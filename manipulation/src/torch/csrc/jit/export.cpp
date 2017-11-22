#include "torch/csrc/jit/export.h"
#include "onnx.pb.h"

#include <ATen/ATen.h>

#include <fstream>

namespace torch { namespace jit {

namespace {

std::string value_name(Value* n) {
  return n->uniqueName();
}


void encodeGraph(onnx::GraphProto * p_g, const std::shared_ptr<Graph> & g);

void encodeTensor(onnx::TensorProto * p, const at::Tensor & tensor) {
  for(auto d : tensor.sizes()) {
    p->add_dims(d);
  }
  at::ScalarType at_type;
  onnx::TensorProto_DataType onnx_type;
  switch(tensor.type().scalarType()) {
    case at::kDouble:
      onnx_type = onnx::TensorProto_DataType_DOUBLE;
      at_type = at::kDouble;
      break;
    case at::kFloat:
      onnx_type = onnx::TensorProto_DataType_FLOAT;
      at_type = at::kFloat;
      break;
    case at::kHalf:
      onnx_type = onnx::TensorProto_DataType_FLOAT16;
      at_type = at::kHalf;
      break;
    case at::kByte:
    case at::kChar:
      onnx_type = onnx::TensorProto_DataType_INT8;
      at_type = at::kByte;
      break;
    case at::kShort:
      onnx_type = onnx::TensorProto_DataType_INT16;
      at_type = at::kShort;
      break;
    case at::kInt:
      onnx_type = onnx::TensorProto_DataType_INT32;
      at_type = at::kInt;
      break;
    case at::kLong:
      onnx_type = onnx::TensorProto_DataType_INT64;
      at_type = at::kLong;
      break;
    default:
      torch::barf("unexpected tensor scalar type");
      break;
  }
  p->set_data_type(onnx_type);
  at::Tensor cont = tensor.toType(at::CPU(at_type)).contiguous();
  p->set_raw_data(cont.data_ptr(), cont.type().elementSizeInBytes() * cont.numel());
}

void addAttribute(onnx::NodeProto * n_p, jit::Node * n, jit::Symbol name) {
  auto attr = n_p->add_attribute();
  attr->set_name(jit::symbolToString(name));
  switch(n->kindOf(name)) {
    case AttributeKind::f:
      attr->set_f(n->f(name));
      attr->set_type(onnx::AttributeProto_AttributeType_FLOAT);
      break;
    case AttributeKind::fs:
      attr->set_type(onnx::AttributeProto_AttributeType_FLOATS);
      for(auto & v : n->fs(name))
        attr->add_floats(v);
      break;
    case AttributeKind::i:
      attr->set_type(onnx::AttributeProto_AttributeType_INT);
      attr->set_i(n->i(name));
      break;
    case AttributeKind::is:
      attr->set_type(onnx::AttributeProto_AttributeType_INTS);
      for(auto & v : n->is(name))
        attr->add_ints(v);
      break;
    case AttributeKind::s:
      attr->set_type(onnx::AttributeProto_AttributeType_STRING);
      attr->set_s(n->s(name));
      break;
    case AttributeKind::ss:
      attr->set_type(onnx::AttributeProto_AttributeType_STRINGS);
      for(auto & v : n->ss(name))
        attr->add_strings(v);
      break;
    case AttributeKind::t: {
      attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
      auto t = attr->mutable_t();
      encodeTensor(t, n->t(name));
    } break;
    case AttributeKind::ts:
      attr->set_type(onnx::AttributeProto_AttributeType_TENSORS);
      for(auto & v : n->ts(name)) {
        auto t = attr->add_tensors();
        encodeTensor(t, v);
      }
      break;
    case AttributeKind::g: {
      attr->set_type(onnx::AttributeProto_AttributeType_GRAPH);
      auto g = attr->mutable_g();
      encodeGraph(g, n->g(name));
    } break;
    case AttributeKind::gs:
      attr->set_type(onnx::AttributeProto_AttributeType_GRAPHS);
      for(auto & v : n->gs(name)) {
        auto g = attr->add_graphs();
        encodeGraph(g, v);
      }
      break;
  }
}

const Dimension* dd = nullptr;

void encodeTypeProtoTensorType(onnx::TypeProto_TensorTypeProto* tensor_type, Value* n) {
  onnx::TypeProto_TensorShapeProto* shape = tensor_type->mutable_shape();
  TensorType* node_type = n->type().get();
  const std::vector<Dimension>& dims = node_type->sizes();
  for (const Dimension& d : dims) {
    dd = &d;
    auto dim = shape->add_dim();
    if (d.is_int) {
      dim->set_dim_value(d.dim);
    } else {
      dim->set_dim_param(d.param);
    }
  }
  onnx::TensorProto_DataType onnx_type;
  switch(node_type->scalarType()) {
    case at::kDouble:
      onnx_type = onnx::TensorProto_DataType_DOUBLE;
      break;
    case at::kFloat:
      onnx_type = onnx::TensorProto_DataType_FLOAT;
      break;
    case at::kHalf:
      onnx_type = onnx::TensorProto_DataType_FLOAT16;
      break;
    case at::kByte:
    case at::kChar:
      onnx_type = onnx::TensorProto_DataType_INT8;
      break;
    case at::kShort:
      onnx_type = onnx::TensorProto_DataType_INT16;
      break;
    case at::kInt:
      onnx_type = onnx::TensorProto_DataType_INT32;
      break;
    case at::kLong:
      onnx_type = onnx::TensorProto_DataType_INT64;
      break;
    default:
      torch::barf("unexpected tensor scalar type");
      break;
  }
  tensor_type->set_elem_type(onnx_type);
}

void encodeValueInfo(onnx::ValueInfoProto* v, Value* n) {
  v->set_name(value_name(n));
  onnx::TypeProto* t = v->mutable_type();
  onnx::TypeProto_TensorTypeProto* tensor_type = t->mutable_tensor_type();
  if (n->hasType()) {
    encodeTypeProtoTensorType(tensor_type, n);
  }
}

void encodeGraph(onnx::GraphProto * p_g, const std::shared_ptr<Graph> & g) {
  JIT_ASSERT(p_g != nullptr);
  p_g->set_name("torch-jit-export");

  for (auto input : g->inputs()) {
    onnx::ValueInfoProto* v = p_g->add_input();
    encodeValueInfo(v, input);
  }
  for (auto output : g->outputs()) {
    onnx::ValueInfoProto* v = p_g->add_output();
    encodeValueInfo(v, output);
  }
  for (auto node : g->nodes()) {
    if (node->kind() == kUndefined && !node->hasUses()) {
      // Undefined nodes never show up in ONNX; they're just a tool
      // to help symbolics do the right thing.
      continue;
    }
    auto p_n = p_g->add_node();
    for(auto input : node->inputs()) {
      p_n->add_input(value_name(input));
    }
    for(auto output : node->outputs()) {
      p_n->add_output(value_name(output));
    }
    p_n->set_op_type(symbolToString(node->kind()));
    for(auto attr_name : node->attributeNames()) {
      addAttribute(p_n, node, attr_name);
    }
  }
  auto num_initializers = g->initializers().size();

  for (int i = 0; i < g->initializers().size(); i++) {
    auto p = p_g->add_initializer();
    p->set_name(g->initializer_names()[i]);
    encodeTensor(p, g->initializers()[i]);
  }
}

void encodeModel(onnx::ModelProto* p_m, const std::shared_ptr<Graph>& g) {
  onnx::GraphProto* p_g = p_m->mutable_graph();
  encodeGraph(p_g, g);
}

}

std::string ExportGraph(const std::shared_ptr<Graph>& graph) {

  onnx::ModelProto model_proto;
  encodeModel(&model_proto, graph);

  std::string out;
  model_proto.SerializeToString(&out);
  return out;
}

}}

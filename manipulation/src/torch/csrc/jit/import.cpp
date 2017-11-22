#include "torch/csrc/jit/import.h"

namespace torch { namespace jit {

std::unique_ptr<Graph> graphProtoToGraph(const onnx::GraphProto& gp);

at::ScalarType protoTypeToATenType(onnx::TensorProto_DataType t) {
  switch(t) {
  case onnx::TensorProto_DataType_FLOAT:
    return at::kFloat;
  case onnx::TensorProto_DataType_INT8:
    return at::kByte;
  case onnx::TensorProto_DataType_INT16:
    return at::kShort;
  case onnx::TensorProto_DataType_INT32:
    return at::kInt;
  case onnx::TensorProto_DataType_INT64:
    return at::kLong;
  case onnx::TensorProto_DataType_FLOAT16:
    return at::kHalf;
  case onnx::TensorProto_DataType_DOUBLE:
    return at::kDouble;
  case onnx::TensorProto_DataType_UNDEFINED:
  case onnx::TensorProto_DataType_UINT8:
  case onnx::TensorProto_DataType_UINT16:
  case onnx::TensorProto_DataType_STRING:
  case onnx::TensorProto_DataType_BOOL:
  case onnx::TensorProto_DataType_UINT32:
  case onnx::TensorProto_DataType_UINT64:
  case onnx::TensorProto_DataType_COMPLEX64:
  case onnx::TensorProto_DataType_COMPLEX128:
    abort();
  }
}

std::pair<at::Tensor, std::string> tensorProtoToTensor(const onnx::TensorProto & tp) {
  at::ScalarType type = protoTypeToATenType(tp.data_type());

  std::vector<int64_t> dims;
  for (int i = 0; i < tp.dims_size(); i++) {
    dims.push_back(tp.dims(i));
  }

  auto data = (void *) tp.raw_data().c_str();

  at::Tensor ret(at::CPU(type).tensorFromBlob(data, dims));
  std::string name;
  if (tp.has_name()) {
    name = tp.name();
  }
  return std::make_pair(ret, name);
}

void convertAttribute(const onnx::AttributeProto & ap, jit::Node & n) {
  Symbol sym = stringToSymbol(ap.name());
  switch(ap.type()) {
  case onnx::AttributeProto_AttributeType_FLOAT:
    n.f_(sym, ap.f());
    break;
  case onnx::AttributeProto_AttributeType_FLOATS: {
    std::vector<double> floats;
    for (int i = 0; i < ap.floats_size(); i++) {
      floats.push_back(ap.floats(i));
    }
    n.fs_(sym, std::move(floats));
    break;
  }
  case onnx::AttributeProto_AttributeType_INT:
    n.i_(sym, ap.i());
    break;
  case onnx::AttributeProto_AttributeType_INTS: {
    std::vector<int64_t> ints;
    for (int i = 0; i < ap.ints_size(); i++) {
      ints.push_back(ap.ints(i));
    }
    n.is_(sym, std::move(ints));
    break;
  }
  case onnx::AttributeProto_AttributeType_STRING:
    n.s_(sym, ap.s());
    break;
  case onnx::AttributeProto_AttributeType_STRINGS: {
    std::vector<std::string> strings;
    for (int i = 0; i < ap.strings_size(); i++) {
      strings.push_back(ap.strings(i));
    }
    n.ss_(sym, std::move(strings));
    break;
  }
  case onnx::AttributeProto_AttributeType_TENSOR:
    n.t_(sym, tensorProtoToTensor(ap.t()).first);
    break;
  case onnx::AttributeProto_AttributeType_TENSORS: {
    std::vector<at::Tensor> tensors;
    for (int i = 0; i < ap.tensors_size(); i++) {
      tensors.push_back(tensorProtoToTensor(ap.tensors(i)).first);
    }
    n.ts_(sym, std::move(tensors));
    break;
  }
  case onnx::AttributeProto_AttributeType_GRAPH:
    n.g_(sym, graphProtoToGraph(ap.g()));
    break;
  case onnx::AttributeProto_AttributeType_GRAPHS: {
    std::vector<std::shared_ptr<Graph>> graphs;
    for (int i = 0; i < ap.graphs_size(); i++) {
      graphs.push_back(graphProtoToGraph(ap.graphs(i)));
    }
    n.gs_(sym, std::move(graphs));
    break;
  }
  case onnx::AttributeProto_AttributeType_UNDEFINED:
    abort();
    break;
  }
}

void convertAttributes(onnx::NodeProto & np, jit::Node & n) {
  for (int i = 0; i < np.attribute_size(); i++) {
    convertAttribute(np.attribute(i), n);
  }
}

std::vector<jit::Dimension> tensorShapeProtoToDimensions(const onnx::TypeProto_TensorShapeProto & tsp) {
  std::vector<jit::Dimension> dims;
  for (int i = 0; i < tsp.dim_size(); i++) {
    if (tsp.dim(i).has_dim_value()) {
      dims.push_back(jit::Dimension(true, tsp.dim(i).dim_value(), ""));
    } else {
      dims.push_back(jit::Dimension(false, -1, tsp.dim(i).dim_param()));
    }
  }
  return dims;
}

TypePtr TypeProtoToTypePtr(const onnx::TypeProto & tp) {
  auto tt = tp.tensor_type();
  auto sh = tensorShapeProtoToDimensions(tt.shape());
  auto et = protoTypeToATenType(tt.elem_type());
  return std::make_shared<TensorType>(TensorType(et, sh));
}

Graph* gg = nullptr;

std::unique_ptr<Graph> graphProtoToGraph(const onnx::GraphProto& gp) {
  std::unique_ptr<Graph> g(new Graph());

  gg = g.get();

  // Values are created (as in `new Value(..)`) by the Node that
  // outputs them. Therefore we initialize the Nodes and Values in
  // several stages.
  //
  // 1) add all input (to the graph) Values, owned by the sentinel Param node
  // 2) add all Nodes and their output Values, but don't intialize inputs
  // 3) initialize inputs of all Nodes
  // 4) initialize inputs of the Return sentinel node
  // 5) fill in type info for graph outputs, and register them as outputs
  // 5) fill in type info for Values from the value_info list in the graph

  // In ONNX proto land, Values are just strings. We are going to make
  // objects out of them, and equal strings must be mapped to the same
  // Value object.
  std::unordered_map<std::string, Value*> value_by_name_of;

  // We initialize Node inputs in a separate pass from the Nodes
  // themselves. To do so, we need to have access to the names of the
  // inputs.
  std::unordered_map<Node*, std::vector<std::string>> inputs_by_node;

  for (int i = 0; i < gp.input_size(); i++) {
    auto vip = gp.input(i);
    auto v = g->addInput();
    v->setType(TypeProtoToTypePtr(vip.type()));
    v->setUniqueName(vip.name());
    value_by_name_of[vip.name()] = v;
  }

  for (int i = 0; i < gp.node_size(); i++) {
    auto np = gp.node(i);
    auto n = g->create(stringToSymbol(np.op_type()), /* num_outputs = */ np.output_size());
    g->appendNode(n);
    for (int j = 0; j < np.output_size(); j++) {
      auto out = n->outputs()[j];
      // we don't know the type here, so that's done in a later pass
      out->setUniqueName(np.output(j));
      value_by_name_of[np.output(j)] = out;
    }
    convertAttributes(np, *n);
    std::vector<std::string> inputs;
    for (int i = 0; i < np.input_size(); i++) {
      inputs.push_back(np.input(i));
      inputs_by_node[n] = inputs;
    }
  }

  for (auto n : g->nodes()) {
    auto search = inputs_by_node.find(n);
    if (search == inputs_by_node.end()) {
      continue;
    }
    for (auto input : search->second) {
      n->addInput(value_by_name_of[input]);
    }
  }

  for (int i = 0; i < gp.output_size(); i++) {
    value_by_name_of[gp.output(i).name()]->setType(TypeProtoToTypePtr(gp.output(i).type()));
    g->registerOutput(value_by_name_of[gp.output(i).name()]);
  }

  for (int i = 0; i < gp.value_info_size(); i++) {
    value_by_name_of[gp.value_info(i).name()]->setType(TypeProtoToTypePtr(gp.value_info(i).type()));
  }

  for (int i = 0; i < gp.initializer_size(); i++) {
    auto init = tensorProtoToTensor(gp.initializer(i));
    g->addInitializer(init.first, init.second);
  }

  return g;
}

std::unique_ptr<Graph> ImportModel(const onnx::ModelProto& mp) {
  return graphProtoToGraph(mp.graph());
}

}}

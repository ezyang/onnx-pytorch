#pragma once

#include "torch/csrc/jit/ir.h"
#include "onnx.pb.h"

namespace torch { namespace jit {

std::unique_ptr<Graph> ImportModel(const onnx::ModelProto& mp);

}}

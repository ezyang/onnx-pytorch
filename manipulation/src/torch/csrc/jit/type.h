#pragma once

#include "torch/csrc/jit/interned_strings.h"
#include "torch/csrc/jit/generic_if.h"
#include "torch/csrc/assertions.h"

#include <ATen/ATen.h>

#include <memory>
#include <iostream>

namespace torch { namespace jit {

struct Dimension {
  Dimension() = delete;

  Dimension(bool is_int, int64_t dim, std::string param)
    : is_int(is_int), dim(dim), param(param)
  { }

  bool is_int;
  int64_t dim;
  std::string param;
};

inline std::vector<Dimension> sizeToDimensions(at::ArrayRef<int64_t> size) {
  std::vector<Dimension> dims;
  for (auto s : size) {
    dims.push_back(Dimension(true, s, ""));
  }
  return dims;
}

// This node represents a single Tensor value
struct TensorType {
  TensorType(const at::Tensor& tensor)
    : scalar_type_(tensor.type().scalarType())
    , sizes_(sizeToDimensions(tensor.sizes())) {}
  TensorType(at::ScalarType scalar_type, std::vector<Dimension> sizes)
    : scalar_type_(scalar_type)
    , sizes_(sizes)
    {}

  at::ScalarType scalarType() const { return scalar_type_; }
  const std::vector<Dimension>& sizes() const { return sizes_; }

private:
  at::ScalarType scalar_type_;
  std::vector<Dimension> sizes_;
};

using TypePtr = std::shared_ptr<TensorType>;

std::ostream& operator<<(std::ostream & out, const TensorType & t);

}} // namespace torch::jit

#include "torch/csrc/jit/type.h"

#include <iostream>

namespace torch { namespace jit {

std::ostream& operator<<(std::ostream & out, const TensorType & t) {
//  TYPE_IF(&t, HandleType)
//    out << "Handle";
//  TYPE_ELSEIF(TensorType)
//    out << at::toString(value->scalarType()) << "(";
//    auto& sizes = value->sizes();
//    for (size_t i = 0; i < sizes.size(); i++) {
//      if (i > 0) {
//        out << ", ";
//      }
//      out << sizes[i];
//    }
//    out << ")";
//  TYPE_END()
  return out;
}

}} // namespace torch::jit

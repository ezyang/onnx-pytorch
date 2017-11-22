add_library (opt-onnx
            torch/csrc/assertions.cpp
            torch/csrc/jit/export.cpp
            torch/csrc/jit/import.cpp
            torch/csrc/jit/interned_strings.cpp
            torch/csrc/jit/ir.cpp
            torch/csrc/jit/type.cpp
            onnx.pb.cc)

add_executable(opt-onnx-roundtrip
            torch/csrc/jit/roundtrip.cpp
            torch/csrc/assertions.cpp
            torch/csrc/jit/export.cpp
            torch/csrc/jit/import.cpp
            torch/csrc/jit/interned_strings.cpp
            torch/csrc/jit/ir.cpp
            torch/csrc/jit/type.cpp
            onnx.pb.cc)

set(CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

target_include_directories(opt-onnx PUBLIC ${CMAKE_SOURCE_DIR}/aten/src)
target_include_directories(opt-onnx PUBLIC ${CMAKE_BINARY_DIR}/aten/src/ATen)
target_include_directories(opt-onnx PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(     opt-onnx PUBLIC ATen protobuf)

target_include_directories(opt-onnx-roundtrip PUBLIC ${CMAKE_SOURCE_DIR}/aten/src)
target_include_directories(opt-onnx-roundtrip PUBLIC ${CMAKE_BINARY_DIR}/aten/src/ATen)
target_include_directories(opt-onnx-roundtrip PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(     opt-onnx-roundtrip PUBLIC ATen protobuf)

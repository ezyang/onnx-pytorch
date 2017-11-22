#include <fstream>

#include "export.h"
#include "import.h"


// reads a protobuf file on stdin and writes it back to stdout.
// doc_strings are lost, but otherwise should be unchanged.
int main(int argc, char** argv) {
  std::stringstream buffer;
  buffer << std::cin.rdbuf();

  onnx::ModelProto mp;
  mp.ParseFromString(buffer.str());
  std::shared_ptr<torch::jit::Graph> g = torch::jit::ImportModel(mp);

  std::string s = torch::jit::ExportGraph(g);
  std::cout << s;

  return 0;
}

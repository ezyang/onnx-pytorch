from subprocess import Popen, PIPE
import sys

import torch
import onnx
from torch.autograd import Variable

def roundtrip_helper(roundtrip_binary, input_proto_path, output_proto_path):
    rt = Popen("{} < {} > {}".format(roundtrip_binary, input_proto_path, output_proto_path), shell=True)
    assert rt.wait() == 0

    input_model = onnx.load(input_proto_path)
    output_model = onnx.load(output_proto_path)

    assert onnx.helper.printable_graph(input_model.graph) \
        == onnx.helper.printable_graph(output_model.graph)

def test_torch_model(roundtrip_binary):
    model=torch.nn.Sequential(torch.nn.Linear(3,2), torch.nn.Linear(2,1))
    x = Variable(torch.randn(10, 3))
    input_path = "/tmp/model.onnx"
    output_path = "/tmp/out.onnx"
    torch.onnx.export(model, x, input_path)
    roundtrip_helper(roundtrip_binary, input_path, output_path)

# invoke as
#   ./test.py /path/to/roundtrip /path/to/model.onnx /path/to/output.onnx
if __name__ == '__main__':
    roundtrip_helper(*sys.argv[1:4])

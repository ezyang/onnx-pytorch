from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import itertools

import torch.jit
from torch.autograd import Variable

import onnx
import onnx_caffe2.backend as c2


torch.set_default_tensor_type('torch.FloatTensor')
try:
    import torch
except ImportError:
    print('Cannot import torch, hence caffe2-torch test will not run.')
    sys.exit(0)


def test_embed_params(proto, model, input, state_dict=None, use_gpu=True):
    """
    This is only a helper debug function so we can test embed_params=False
    case as well on pytorch front
    This should likely be removed from the release version of the code
    """
    graph_def = onnx.GraphProto.FromString(proto)
    onnx.checker.check_graph(graph_def)

    # Translate the parameters into Caffe2 form
    W = {}
    if state_dict:
        parameters = []
        # Passed in state_dict may have a different order.  Make
        # sure our order is consistent with the model's order.
        # TODO: Even better: keyword arguments!
        for k in model.state_dict():
            parameters.append(state_dict[k])
    else:
        parameters = model.state_dict().values()

    for k, v in zip(graph_def.input, itertools.chain(parameters,
                                                     list(torch.jit.flatten(input)))):
        if isinstance(v, Variable):
            W[k] = v.data.cpu().numpy()
        else:
            W[k] = v.cpu().numpy()

    caffe2_out_workspace = c2.run_graph(
        init_graph=None,
        predict_graph=graph_def,
        inputs=W,
        use_gpu=use_gpu)
    caffe2_out = caffe2_out_workspace[0]
    return caffe2_out

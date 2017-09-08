import torch
import torch.jit
from torch.autograd import Variable
from test_common import TestCase, run_tests, skipIfNoLapack

from model_defs.alexnet import AlexNet
from model_defs.mnist import MNIST
from model_defs.word_language_model import RNNModel
from model_defs.vgg import *
from model_defs.resnet import Bottleneck, ResNet
from model_defs.inception import Inception3
from model_defs.squeezenet import SqueezeNet
from model_defs.super_resolution import SuperResolutionNet
from model_defs.densenet import DenseNet
from model_defs.srresnet import SRResNet
from model_defs.dcgan import _netD, _netG, weights_init, bsz, imgsz, nz
from model_defs.op_test import DummyNet, ConcatNet, PermuteNet, PReluNet

import onnx
import google.protobuf.text_format

torch.set_default_tensor_type('torch.FloatTensor')

if torch.cuda.is_available():
    def toC(x):
        return x.cuda()
else:
    def toC(x):
        return x

BATCH_SIZE = 2


class TestModels(TestCase):
    maxDiff = None

    def assertONNXExpected(self, binary_pb, subname=None):
        graph_def = onnx.GraphProto.FromString(binary_pb)
        self.assertExpected(google.protobuf.text_format.MessageToString(graph_def, float_format='.15g'), subname)

    def test_ops(self):
        x = Variable(
            torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0), volatile=True
        )
        trace, _ = torch.jit.record_trace(toC(DummyNet()), toC(x))
        self.assertExpected(str(trace))
        self.assertONNXExpected(trace.export(False), "pbtxt")

    def test_prelu(self):
        x = Variable(
            torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0), volatile=True
        )
        trace, _ = torch.jit.record_trace(PReluNet(), x)
        self.assertExpected(str(trace))
        self.assertONNXExpected(trace.export(False), "pbtxt")

    def test_concat(self):
        input_a = Variable(torch.randn(BATCH_SIZE, 3), volatile=True)
        input_b = Variable(torch.randn(BATCH_SIZE, 3), volatile=True)
        inputs = (toC(input_a), toC(input_b))
        trace, _ = torch.jit.record_trace(toC(ConcatNet()), inputs, num_derivatives=0)
        self.assertExpected(str(trace))
        self.assertONNXExpected(trace.export(False), "pbtxt")

    def test_permute(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 10, 12), volatile=True)
        trace, _ = torch.jit.record_trace(PermuteNet(), x)
        self.assertExpected(str(trace))
        self.assertONNXExpected(trace.export(False), "pbtxt")

    def test_srresnet(self):
        x = Variable(torch.randn(1, 3, 224, 224).fill_(1.0),
                     volatile=True)
        trace, _ = torch.jit.record_trace(
            toC(SRResNet(rescale_factor=4, n_filters=64, n_blocks=8)), toC(x))
        self.assertExpected(str(trace))
        self.assertONNXExpected(trace.export(False), "pbtxt")

    @skipIfNoLapack
    def test_super_resolution(self):
        x = Variable(
            torch.randn(BATCH_SIZE, 1, 224, 224).fill_(1.0), volatile=True
        )
        trace, _ = torch.jit.record_trace(
            toC(SuperResolutionNet(upscale_factor=3)), toC(x))
        self.assertExpected(str(trace))
        self.assertONNXExpected(trace.export(False), "pbtxt")

    def test_alexnet(self):
        x = Variable(
            torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0), volatile=True
        )
        trace, _ = torch.jit.record_trace(toC(AlexNet()), toC(x))
        self.assertExpected(str(trace))
        self.assertONNXExpected(trace.export(False), "pbtxt")

    def test_mnist(self):
        x = Variable(torch.randn(BATCH_SIZE, 1, 28, 28).fill_(1.0),
                     volatile=True)
        trace, _ = torch.jit.record_trace(toC(MNIST()), toC(x))
        self.assertExpected(str(trace))
        # FeatureDropout symbolic is not implemented yet
        # self.assertONNXExpected(trace.export(False), "pbtxt")

    def test_vgg(self):
        # VGG 16-layer model (configuration "D")
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0),
                     volatile=True)
        vgg16 = make_vgg16()
        trace, _ = torch.jit.record_trace(toC(vgg16), toC(x))
        self.assertExpected(str(trace), "16")
        self.assertONNXExpected(trace.export(False), "16-pbtxt")

        # VGG 16-layer model (configuration "D") with batch normalization
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0),
                     volatile=True)
        vgg16_bn = make_vgg16_bn()
        trace, _ = torch.jit.record_trace(toC(vgg16_bn), toC(x))
        self.assertExpected(str(trace), "16_bn")
        self.assertONNXExpected(trace.export(False), "16_bn-pbtxt")

        # VGG 19-layer model (configuration "E")
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0),
                     volatile=True)
        vgg19 = make_vgg19()
        trace, _ = torch.jit.record_trace(toC(vgg19), toC(x))
        self.assertExpected(str(trace), "19")
        self.assertONNXExpected(trace.export(False), "19-pbtxt")

        # VGG 19-layer model (configuration 'E') with batch normalization
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0),
                     volatile=True)
        vgg19_bn = make_vgg19_bn()
        trace, _ = torch.jit.record_trace(toC(vgg19_bn), toC(x))
        self.assertExpected(str(trace), "19_bn")
        self.assertONNXExpected(trace.export(False), "19_bn-pbtxt")

    def test_resnet(self):
        # ResNet50 model
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0),
                     volatile=True)
        resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        trace, _ = torch.jit.record_trace(toC(resnet50), toC(x))
        self.assertExpected(str(trace), "50")
        self.assertONNXExpected(trace.export(False), "50-pbtxt")

    def test_inception(self):
        x = Variable(
            torch.randn(BATCH_SIZE, 3, 299, 299).fill_(1.0), volatile=True)
        trace, _ = torch.jit.record_trace(toC(Inception3()), toC(x))
        self.assertExpected(str(trace), "3")
        self.assertONNXExpected(trace.export(False), "3-pbtxt")

    def test_squeezenet(self):
        # SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and
        # <0.5MB model size
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0),
                     volatile=True)
        sqnet_v1_0 = SqueezeNet(version=1.1)
        trace, _ = torch.jit.record_trace(toC(sqnet_v1_0), toC(x))
        self.assertExpected(str(trace), "1_0")
        self.assertONNXExpected(trace.export(False), "1_0-pbtxt")

        # SqueezeNet 1.1 has 2.4x less computation and slightly fewer params
        # than SqueezeNet 1.0, without sacrificing accuracy.
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0),
                     volatile=True)
        sqnet_v1_1 = SqueezeNet(version=1.1)
        trace, _ = torch.jit.record_trace(toC(sqnet_v1_1), toC(x))
        self.assertExpected(str(trace), "1_1")
        self.assertONNXExpected(trace.export(False), "1_1-pbtxt")

    def test_densenet(self):
        # Densenet-121 model
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0),
                     volatile=True)
        dense121 = DenseNet(num_init_features=64, growth_rate=32,
                            block_config=(6, 12, 24, 16))
        trace, _ = torch.jit.record_trace(toC(dense121), toC(x))
        self.assertExpected(str(trace), "121")
        self.assertONNXExpected(trace.export(False), "121-pbtxt")

    def test_dcgan(self):
        # note, could have more than 1 gpu
        netG = _netG(1)
        netG.apply(weights_init)
        netD = _netD(1)
        netD.apply(weights_init)

        input = torch.Tensor(bsz, 3, imgsz, imgsz)
        noise = torch.Tensor(bsz, nz, 1, 1)
        fixed_noise = torch.Tensor(bsz, nz, 1, 1).normal_(0, 1)

        fixed_noise = Variable(fixed_noise)

        netD.zero_grad()
        inputv = Variable(input)
        trace, _ = torch.jit.record_trace(toC(netD), toC(inputv))
        self.assertExpected(str(trace), "dcgan-netD")
        self.assertONNXExpected(trace.export(False), "dcgan-netD-pbtxt")

        noise.resize_(bsz, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        trace, _ = torch.jit.record_trace(toC(netG), toC(noisev))
        self.assertExpected(str(trace), "dcgan-netG")
        self.assertONNXExpected(trace.export(False), "dcgan-netG-pbtxt")

    def run_word_language_model(self, model_name):
        # Args:
        #   model: string, one of RNN_TANH, RNN_RELU, LSTM, GRU
        #   ntokens: int, len(corpus.dictionary)
        #   emsize: int, default 200, size of embedding
        #   nhid: int, default 200, number of hidden units per layer
        #   nlayers: int, default 2
        #   dropout: float, default 0.5
        #   tied: bool, default False
        #   batchsize: int, default 2
        ntokens = 10
        emsize = 5
        nhid = 5
        nlayers = 5
        dropout = 0.2
        tied = False
        batchsize = 5
        model = RNNModel(model_name, ntokens, emsize,
                         nhid, nlayers, dropout,
                         tied, batchsize)
        x = Variable(torch.LongTensor(10, batchsize).fill_(1),
                     volatile=False)
        trace, _ = torch.jit.record_trace(model, x, model.hidden)
        self.assertExpected(str(trace))
        self.assertONNXExpected(trace.export(False), "pbtxt")

    def test_word_language_model_RNN_TANH(self):
        model_name = 'RNN_TANH'
        self.run_word_language_model(model_name)

    def test_word_language_model_RNN_RELU(self):
        model_name = 'RNN_RELU'
        self.run_word_language_model(model_name)

    def test_word_language_model_LSTM(self):
        model_name = 'LSTM'
        self.run_word_language_model(model_name)

    def test_word_language_model_GRU(self):
        model_name = 'GRU'
        self.run_word_language_model(model_name)


if __name__ == '__main__':
    run_tests()

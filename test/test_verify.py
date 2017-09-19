import torch
from torch.autograd import Variable, Function
from torch.nn import Module, Parameter
import onnx_caffe2.backend as backend
from onnx_pytorch.verify import verify

from test_common import TestCase, run_tests


class TestVerify(TestCase):
    maxDiff = None

    def assertVerifyExpectFail(self, *args, **kwargs):
        try:
            verify(*args, **kwargs)
        except AssertionError as e:
            if str(e):
                self.assertExpected(str(e))
                return
            else:
                raise
        # Don't put this in the try block; the AssertionError will catch it
        self.assertTrue(False, msg="verify() did not fail when expected to")

    def test_result_different(self):
        class BrokenAdd(Function):
            @staticmethod
            def symbolic(g, a, b):
                return g.appendNode(g.create("Add", [a, b]))

            @staticmethod
            def forward(ctx, a, b):
                return a.sub(b) # yahaha! you found me!

        class MyModel(Module):
            def forward(self, x, y):
                return BrokenAdd().apply(x, y)

        x = Variable(torch.Tensor([1,2]))
        y = Variable(torch.Tensor([3,4]))
        self.assertVerifyExpectFail(MyModel(), (x, y), backend)


    def test_jumbled_params(self):
        class MyModel(Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, x):
                y = x * x
                self.param = Parameter(torch.Tensor([2]))
                return y

        x = Variable(torch.Tensor([1,2]))
        self.assertVerifyExpectFail(MyModel(), x, backend)


    def test_modifying_params(self):
        class MyModel(Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.param = Parameter(torch.Tensor([2]))

            def forward(self, x):
                y = x * x
                self.param.data.add_(1.0)
                return y

        x = Variable(torch.Tensor([1,2]))
        self.assertVerifyExpectFail(MyModel(), x, backend)


    def test_dynamic_model_structure(self):
        class MyModel(Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.iters = 0

            def forward(self, x):
                if self.iters % 2 == 0:
                    r = x * x
                else:
                    r = x + x
                self.iters += 1
                return r

        x = Variable(torch.Tensor([1,2]))
        self.assertVerifyExpectFail(MyModel(), x, backend)


    def test_embedded_constant_difference(self):
        class MyModel(Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.iters = 0

            def forward(self, x):
                r = x[self.iters % 2]
                self.iters += 1
                return r

        x = Variable(torch.Tensor([[1,2], [3,4]]))
        self.assertVerifyExpectFail(MyModel(), x, backend)


    def test_explicit_test_args(self):
        class MyModel(Module):
            def forward(self, x):
                if x.data.sum() == 1.0:
                    return x + x
                else:
                    return x * x

        x = Variable(torch.Tensor([[6,2]]))
        y = Variable(torch.Tensor([[2,-1]]))
        self.assertVerifyExpectFail(MyModel(), x, backend, test_args=[(y,)])


if __name__ == '__main__':
    run_tests()

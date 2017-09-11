#!/bin/sh
git subtree pull --prefix pytorch https://github.com/pytorch/pytorch.git master --squash
git subtree pull --prefix onnx https://github.com/onnx/onnx.git master --squash
git subtree pull --prefix onnx-caffe2 https://github.com/onnx/onnx-caffe2.git master --squash

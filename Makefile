.PHONY: all onnx onnx-caffe2 pytorch
all: onnx onnx-caffe2 pytorch
onnx:
	cd onnx && python setup.py develop
onnx-caffe2:
	cd onnx-caffe2 && python setup.py develop
pytorch:
	cd pytorch && python setup.py build develop
pytorch-develop:
	cd pytorch && python setup.py develop

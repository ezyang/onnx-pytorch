.PHONY: all onnx onnx-caffe2 pytorch pytorch-onnx
all: onnx onnx-caffe2 pytorch pytorch-onnx
pytorch-onnx:
	python setup.py develop
onnx:
	cd onnx && python setup.py develop
onnx-caffe2:
	cd onnx-caffe2 && python setup.py develop
pytorch:
	cd pytorch && python setup.py build develop
pytorch-develop:
	cd pytorch && python setup.py develop

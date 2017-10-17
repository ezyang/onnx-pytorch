.PHONY: all onnx onnx-caffe2 pytorch pytorch-onnx test
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
test:
	python pytorch/test/test_jit.py
	python test/test_models.py
	python test/test_caffe2.py
	python test/test_verify.py

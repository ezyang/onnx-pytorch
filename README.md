# Support scripts for PyTorch-ONNX

This repository contains end-to-end tests for PyTorch's ONNX support, including
exporting models to Caffe2.

## Developing

To run the Caffe2 tests, you'll need a working install of Caffe2.
If you're on a devgpu, consider getting a copy via `conda install -c ezyang/label/devgpu caffe2`

```
# Only needs to be done once
./install-deps.sh

# These commands must be rerun if the C++ files in these projects change
(cd onnx && python setup.py develop)
(cd onnx-caffe2 && python setup.py develop)
(cd pytorch && python setup.py build develop)
python setup.py develop

# Run tests
python pytorch/test/test_onnx.py
python pytorch/test/test_jit.py
python test/test_models.py
python test/test_caffe2.py
python test/test_verify.py
```

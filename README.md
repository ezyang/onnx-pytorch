# Support scripts for PyTorch-ONNX

This repository contains end-to-end tests for PyTorch's ONNX support, including
exporting models to Caffe2.

## Developing

To run the Caffe2 tests, you'll need a working install of Caffe2.
If you're on a devgpu, consider getting a copy via
1) `conda install -c ezyang/label/devgpu caffe2` for py3,
2) `conda install -c houseroad/label/devgpu caffe2` for py2.7.

```
# Only needs to be done once
./install-deps.sh
make
make test
```

## Troubleshooting

#### `ImportError: libATen.so.1: undefined symbol` after update.

Sometimes after update C++ libraries don't get rebuilt correctly in PyTorch. Try `cd pytorch && python setup.py clean` or `rm -rf pytorch/torch/lib/build/ pytorch/build`.

#### Pending changes with submodules after update

Try doing `git submodule sync && git submodule update --init --recursive` to refresh everything.

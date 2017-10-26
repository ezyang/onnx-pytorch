#!/bin/sh

set -e

conda install -y numpy pyyaml mkl setuptools cmake cffi atlas
conda install -y libstdcxx-ng  # scipy requires newer libstdcxx
conda install -y scipy
conda install -y -c soumith magma-cuda80
conda install -y -c conda-forge protobuf
conda install -y pytest

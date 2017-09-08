#!/usr/bin/env bash

env | grep GIT
env | grep ghprb
env | grep jenkins_python_version
env | grep BUILDER

if [ -z "$github_token" ]; then
    echo "could not find env variable github_token, exiting"
    exit 1
fi

COMMIT_TO_TEST=""
if [ -z "$ghprbActualCommit" ]; then
    COMMIT_TO_TEST=$GIT_COMMIT
else
    # building a pull request
    COMMIT_TO_TEST=$ghprbActualCommit
fi

BRANCH_TO_TEST=""
if [ -z "$ghprbActualCommit" ]; then
    BRANCH_TO_TEST=$GIT_BRANCH
else
    # building a pull request
    BRANCH_TO_TEST="origin/pr/$ghprbPullId/merge"
fi

if [ -z "$jenkins_python_version" ]; then
    echo "jenkins_python_version is not defined. define it to 2 or 3"
    exit 1
fi

BUILDER=${BUILDER_TYPE:-LOCAL}
BUILDER_OS=${BUILDER_OS:-LINUX}

echo "Username: $USER"
echo "Homedir: $HOME"
echo "Home ls:"
ls -alh ~/ || true
echo "Current directory: $(pwd)"
echo "Branch: $GIT_BRANCH"
echo "Commit: $GIT_COMMIT"
echo "OS: $OS"

echo "Disks:"
df -h || true

if [ "$OS" == "LINUX" ]; then
    echo "running nvidia-smi"
    nvidia-smi

    echo "Processor info"
    cat /proc/cpuinfo|grep "model name" | wc -l
    cat /proc/cpuinfo|grep "model name" | sort | uniq
    cat /proc/cpuinfo|grep "flags" | sort | uniq

    echo "Linux release:"
    lsb_release -a || true

else
    echo "Processor info"····
    sysctl -n machdep.cpu.brand_string
fi

uname -a

# See pytorch/builder build_nimbix.sh for ccache and cuda installer

if [ "$OS" == "LINUX" ]; then
    if ! ls ~/ccache/bin/ccache; then
        echo "Please setup ccache on the builder!"
        echo "This can be done by running a regular PyTorch build on it"
        exit 0
    fi
    export PATH=~/ccache/lib:$PATH
    export CUDA_NVCC_EXECUTABLE=~/ccache/cuda/nvcc

    if ! ls /usr/local/cuda-8.0; then
        echo "Please setup CUDA 8.0 on the builder!"
        echo "This can be done by running a regular PyTorch build on it"
        exit 0
    fi

    # add cuda to PATH and LD_LIBRARY_PATH
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


    if ! ls /usr/local/cuda/lib64/libcudnn.so.6.0.21; then
        echo "Please setup CuDNN 6.0.21 on the builder!"
        echo "This can be done by running a regular PyTorch build on it"
        exit 0
    fi
fi

echo "nvcc: $(which nvcc)"

if ! ls ~/miniconda; then
    echo "Please setup miniconda on the builder!"
    echo "This can be done by running a regular PyTorch build on it"
    exit 0
fi
export PATH="$HOME/miniconda/bin:$PATH"

if [ ! -d "$WORKSPACE-env" ]; then
    conda create -p "$WORKSPACE-env" python=$PYTHON_VERSION -y
fi
source activate "$WORKSPACE-env"
export CONDA_ROOT_PREFIX="$WORKSPACE-env"

echo "Conda root: $CONDA_ROOT_PREFIX"

if ! which cmake
then
    conda install -y cmake
fi

# install mkl
conda install -y mkl numpy

if [ "$OS" == "LINUX" ]; then
    conda install -y magma-cuda80 -c soumith
fi

# add mkl to CMAKE_PREFIX_PATH
export CMAKE_LIBRARY_PATH=$CONDA_ROOT_PREFIX/lib:$CONDA_ROOT_PREFIX/include:$CMAKE_LIBRARY_PATH
export CMAKE_PREFIX_PATH=$CONDA_ROOT_PREFIX

echo "Python Version:"
python --version

echo "GCC Version:"
gcc --version

cd $WORKSPACE

echo "Installing onnx"
conda install -y -c ezyang/label/gcc5 -c conda-forge protobuf scipy
(cd onnx && time python setup.py install)
python -c "import onnx"

echo "Installing caffe2"
conda install -y -c ezyang/label/gcc5 -c conda-forge caffe2

echo "Installing onnx-caffe2"
(cd onnx-caffe2 && time python setup.py install)
python -c "import onnx_caffe2"

echo "Installing pytorch"
if [ "$OS" == "OSX" ]; then
    export MACOSX_DEPLOYMENT_TARGET=10.9
    export CC=clang
    export CXX=clang++
fi
(cd pytorch && pip install -r requirements.txt || true)
(cd pytorch && time python setup.py install)

echo "Testing"
time python pytorch/test/test_onnx.py
time python pytorch/test/test_jit.py
time python test/test_models.py
time python test/test_caffe2.py

echo "ALL CHECKS PASSED"

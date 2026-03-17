#!/bin/bash
set -e  # 遇到错误立即退出

# ==========================================
# 1. 自动定位路径逻辑
# ==========================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MXBENCH_INCLUDE_DIR="${PROJECT_ROOT}/mxbench/install/include"
export NVBENCH_INSTALL_PATH="${PROJECT_ROOT}/mxbench/install"
echo "========================================"
echo "脚本位置: ${SCRIPT_DIR}"
echo "定位根目录: ${PROJECT_ROOT}"
echo "========================================"

cd "${PROJECT_ROOT}"

echo ">>> install py packages "
pip3 install cmake==3.30.4 setuptools-scm==8.0 pybind11 build -i https://repo.metax-tech.com/r/pypi/simple


echo ">>> setting mxbench build env"
if [ -f "${PROJECT_ROOT}/mxbench/env.sh" ]; then
    cd "${PROJECT_ROOT}/mxbench"
    source env.sh
else
    echo "[ERROR]  source env.sh script not exit"
    exit 1
fi

echo ">>> building  mxbench c++ performance test..."
if [[ ! -d "${PROJECT_ROOT}/mxbench" ]]; then
    echo "[ERROR] not found mxbench dir"
    exit 1
fi

cd ${PROJECT_ROOT}/mxbench
# 强制清理 C++ build 目录，防止缓存冲突
rm -rf build
mkdir -p build
cd build

cmake_maca -DCMAKE_CXX_STANDARD=17 \
           -DCMAKE_CUDA_STANDARD=17 \
           -DCMAKE_CUDA_ARCHITECTURES=80 \
           -DCMAKE_CUDA_FLAGS="-I${MXBENCH_INCLUDE_DIR}" \
           -DCMAKE_CXX_FLAGS="-Wno-unused-parameter -Wno-error -Wno-implicit-float-conversion -I${MXBENCH_INCLUDE_DIR}" \
           ..

# [4/5] 执行编译
echo ">>> start to build c++ mxbench test"
make_maca VERBOSE=1


echo ">>> mxbench make packaging ..."
cd ../python

echo " cleaning (build, dist, egg-info) build temp files..."
rm -rf build/lib.linux*  build/temp.linux* dist/* *.egg-info

echo " mxbench python package building "
python setup.py bdist_wheel

echo " installing mxbench python package "
pip3 install ./dist/*.whl
echo "========================================"
echo "[SUCCESS] all done"
echo "========================================"
#!/bin/bash
set -e  # 遇到错误立即退出

# ==========================================
# 1. 自动定位路径逻辑
# ==========================================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MXBENCH_INCLUDE_DIR="${PROJECT_ROOT}/mxbench/install/include"

echo "========================================"
echo "脚本位置: ${SCRIPT_DIR}"
echo "定位根目录: ${PROJECT_ROOT}"
echo "========================================"

cd "${PROJECT_ROOT}"

# ==========================================
# 2. 开始环境配置
# ==========================================

if [[ ! -f "setup.py" ]]; then
    echo "[ERROR] 无法定位到 mcoplib 根目录 (未找到 setup.py)。"
    exit 1
fi

# [1/5] 安装 Python 依赖
echo ">>> [1/5] 安装 Python 依赖..."
pip3 install cmake==3.30.4 setuptools-scm==8.0 pybind11 build
echo "[OK] Python 依赖安装完成"

# [2/5] 安装 mcoplib
echo ">>> [2/5] 安装 mcoplib..."
if [ -f "env.sh" ]; then
    source env.sh
else
    echo "[WARNING] 跳过 source env.sh"
fi
pip install -e . --no-build-isolation
echo "[OK] mcoplib 安装完成"

# [3/5] 编译 mxbench C++ 核心
echo ">>> [3/5] 配置并编译 mxbench (C++)..."
if [[ ! -d "mxbench" ]]; then
    echo "[ERROR] 未找到 mxbench 目录。"
    exit 1
fi

cd mxbench
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
echo ">>> [4/5] 开始编译 (make_maca)..."
make_maca VERBOSE=1
echo "[OK] mxbench C++ 编译完成"

# [5/5] 安装 mxbench Python 绑定
echo ">>> [5/5] 安装 mxbench Python 接口..."
cd ../python

# ==========================================
# [新增] 关键修复：清理 Python 侧的旧构建缓存
# ==========================================
echo "    正在清理旧的构建缓存 (build, dist, egg-info)..."
rm -rf build dist *.egg-info

python setup.py develop

echo "========================================"
echo "[SUCCESS] 环境配置全部完成！"
echo "========================================"
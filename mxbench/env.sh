DEFAULT_DIR="/opt/maca"
USER_HOME="$HOME"
echo "cur user home dir:$USER_HOME"

export MACA_PATH=${1:-$DEFAULT_DIR}
export CUDA_PATH=${USER_HOME}/cu-bridge/CUDA_DIR
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export PATH=${CUDA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${CUCC_PATH}/tools:${CUCC_PATH}/bin:$PATH
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export CUCC_CMAKE_ENTRY=2
export CUDNN_INCLUDE_DIR=${MACA_PATH}/tools/cu-bridge/include   # 头文件目录
export CUDNN_LIB_DIR=${MACA_PATH}/tools/cu-bridge               # 库文件目录
export ONNXRUNTIME_HOME=/opt/maca-ai/onnxruntime-maca
export WCUDA_DEBUG=1
export CUDNN_HOME=${MACA_PATH}/tools/cu-bridge
echo "MACA PATH: ${MACA_PATH} Compile Code"
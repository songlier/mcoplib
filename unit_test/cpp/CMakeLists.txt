cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# 设置项目名称和使用的语言（需要包含CUDA）
project(mcoplib_cv_test LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置CUDA架构
set(CUDA_ARCHITECTURES "native")  # 自动检测当前GPU架构

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3 -Xcompiler -fPIC  -Xcompiler=-Wno-unused-parameter -Xcompiler=-std=gnu++17 -Xcompiler=-Wno-error -Xcompiler=-Wno-cast-qual -Xcompiler=-Wno-implicit-float-conversion \
                    -mllvm -metaxgpu-disable-bsm-offset=0 \
                    -mllvm -metaxgpu-combine-intrinsic-bfe=false \
                    -mllvm -metaxgpu-igroup=true \
                    -mllvm -metaxgpu-igroup-config=B8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17 -O2 -Wno-error -Wno-c++17-extensions -Wno-unused-parameter ")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-error -Wno-unused-parameter ")

# --------------------------
# CUDA
# --------------------------
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDAToolkit include dirs = ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "CUDAToolkit libraries = ${CUDAToolkit_LIBRARIES}")


set(MCOPLIB_CV_INSTALL_DIRS /opt/maca-ai/mcoplib)
set(MCOPLIB_CV_INCLUDE_DIRS ${MCOPLIB_CV_INSTALL_DIRS}/include)
set(MCOPLIB_CV_LIB_DIRS ${MCOPLIB_CV_INSTALL_DIRS}/lib)

add_executable(${PROJECT_NAME}_exec
    test_process.cpp
)
#find_package(CUDA REQUIRED)
target_include_directories(${PROJECT_NAME}_exec PRIVATE
    ${MCOPLIB_CV_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS} 
)

target_link_directories(${PROJECT_NAME}_exec PRIVATE ${MCOPLIB_CV_LIB_DIRS})
target_link_libraries(${PROJECT_NAME}_exec PRIVATE CUDA::cudart CUDA::nvrtc)
target_link_libraries(${PROJECT_NAME}_exec PRIVATE mcoplib_cv)

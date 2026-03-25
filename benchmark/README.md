# fused_add_rms_norm 算子性能测试

本目录包含使用 NVBench 测试 `fused_add_rms_norm` 算子性能的代码。

## 文件说明

- `fused_add_rms_norm_nvbench.py`: 完整版本的性能测试代码，支持多种参数组合
- `fused_add_rms_norm_simple.py`: 简化版本的测试代码，用于快速验证
- `README.md`: 本文档

## 环境要求

1. **硬件要求**:
   - NVIDIA GPU（支持 CUDA）
   - 足够的 GPU 内存用于测试

2. **软件要求**:
   - Python >= 3.8
   - PyTorch >= 1.10
   - NVBench Python API (`pynvbench`)
   - mcoplib 算子库

## 安装依赖

### 1. 安装 NVBench

参考 `nvbench/python/README.md` 中的说明：

```bash
cd nvbench/python
pip install -e .
```

### 2. 安装 mcoplib

确保 mcoplib 已经正确编译和安装：

```bash
cd mcoplib
pip install -e .
```

## 使用方法

### 快速开始（简化版本）

运行简化版本的测试：

```bash
cd mcoplib/benchmark
python fused_add_rms_norm_simple.py
```

这将使用固定的参数（1024 tokens, 512 hidden size, float16）运行测试。

### 完整测试

运行完整版本的测试，包含多种参数组合：

```bash
python fused_add_rms_norm_nvbench.py
```

### 常用命令行选项

```bash
# 列出所有可用的 benchmark
python fused_add_rms_norm_nvbench.py --list

# 在特定设备上运行
python fused_add_rms_norm_nvbench.py --devices 0

# 以 JSON 格式输出结果
python fused_add_rms_norm_nvbench.py --json

# 以 Markdown 格式输出结果
python fused_add_rms_norm_nvbench.py --markdown

# 禁用控制台输出（仅输出到文件）
python fused_add_rms_norm_nvbench.py --no-console

# 输出到文件
python fused_add_rms_norm_nvbench.py --output results.json
```

### 指定特定的测试参数

```bash
# 只测试 float16 类型
python fused_add_rms_norm_nvbench.py -a DType=0

# 只测试特定的数据形状
python fused_add_rms_norm_nvbench.py -a NumTokens=1024 -a HiddenSize=512

# 组合多个参数
python fused_add_rms_norm_nvbench.py -a DType=0 -a NumTokens=1024
```

## 输出说明

NVBench 会输出详细的性能指标，包括：

1. **时间指标**:
   - 平均执行时间
   - 最小/最大时间
   - 标准差

2. **吞吐量**:
   - Elements/sec（每秒处理的元素数）
   - Bytes/sec（每秒处理的字节数）
   - Global Memory Throughput（全局内存吞吐量）

3. **其他信息**:
   - GPU 设备信息
   - 测试参数（数据类型、形状等）

## 测试参数说明

### DType（数据类型）

- `0`: float16 (半精度浮点数)
- `1`: float32 (单精度浮点数)
- `2`: bfloat16 (脑浮点数)

### NumTokens（令牌数）

测试的令牌数量，表示批处理大小 × 序列长度。

常用值:
- 小规模: 128, 256
- 中规模: 1024, 2048
- 大规模: 4096, 8192

### HiddenSize（隐藏层大小）

隐藏层的维度大小，对应模型中的隐藏维度。

常用值:
- 小模型: 512, 768
- 中等模型: 1024, 2048
- 大模型: 4096, 8192

## 性能优化建议

根据测试结果，可以考虑以下优化：

1. **数据类型**: 使用 float16 或 bfloat16 可以提高吞吐量
2. **批处理**: 增加 NumTokens 可以提高 GPU 利用率
3. **内存访问**: 优化内存访问模式可以提高性能

## 故障排除

### 导入错误

如果遇到 `ImportError: No module named 'mcoplib._C'`：

```bash
# 确保 mcoplib 已正确安装
cd mcoplib
pip install -e .
```

### NVBench 未安装

如果遇到 `ImportError: No module named 'cuda.bench'`：

```bash
# 安装 NVBench Python API
cd nvbench/python
pip install -e .
```

### CUDA 相关错误

确保：
1. CUDA 驱动已正确安装
2. PyTorch 已正确编译 CUDA 支持
3. GPU 设备可用：`python -c "import torch; print(torch.cuda.is_available())"`

## 参考文档

- [NVBench 文档](../../docs/)
- [PyTorch CUDA 扩展](https://pytorch.org/tutorials/advanced/cpp_extension.html)
- [mcoplib 算子文档](../op/vllm/)

## 许可证

请参考主项目的许可证文件。

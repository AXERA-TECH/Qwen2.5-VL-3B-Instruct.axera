# Qwen2.5-VL-3B-Instruct 模型转换
这个模型分为 Vision Encoder 和 Language Model 两部分，分别进行转换。

## 一、转换 Vision Encoder 

导出 Vision Encoder 为 onnx，然后通过 `pulsar2 build` 转换为 axmodel模型，

### 1. 创建虚拟环境

```
conda create -n qwen2_5_vl python=3.12 -y
conda activate qwen2_5_vl
```

### 2. 安装依赖

```
pip install -r requirements.txt
```
本代码依赖 `transformers>=4.49.0`，最好使用这个版本的`transformers`，其它版本可能会有不兼容问题。

### 3. 导出模型（PyTorch -> ONNX）

在导出onnx之前需要先下从 huggingface 或 model scope 下载模型。这里假设模型的保存目录是 `../Qwen/Qwen2.5-VL-3B-Instruct/`。    

由于视频输入的token数较多，直接导出的onnx会比较大，编译和运行都会比较慢，所以这里按时间步（temporal_patch_size）导出onnx模型。这个模型的视觉部分按时间步运行时，不同时间步的`rotary_pos_emb`, `cu_seqlens`, `cu_window_seqlens`, `window_index`都是相同的，所以导出比较方便。  
和模型原始输入不同的是，这里为了让模型使用UINT8输入，特意将`Qwen2VLImageProcessor` 编排过的 image patches 又转换成了图片的格式（具体代码在[preprocess.py](preprocess.py)里面可以看到）。

可以执行`bash export_video.sh`直接导出模型，以下是详细步骤。  

1). 运行模型，保存导出onnx需要的参数
```
python run_video_by_sec.py ../Qwen/Qwen2.5-VL-3B-Instruct/
```
这里会保存 `hidden_states`, `rotary_pos_emb`, `cu_seqlens`, `cu_window_seqlens`, `window_index`。  
这几个tensor，除了`hidden_states`和像素值、图像尺寸有关外，其它四个都只和图像尺寸相关。所以如果模型的输入尺寸固定，这几个tensor可以固定到onnx模型中。

2). 导出onnx模型
```
python export.py ../Qwen/Qwen2.5-VL-3B-Instruct/  video
```
这一步会生成 `Qwen2.5-VL-3B-Instruct_vision.onnx`和`Qwen2.5-VL-3B-Instruct_vision.onnx.data`,计算图和权重参数分离。

3). 测试onnx模型

```
python test_onnx_video_by_sec.py ../Qwen/Qwen2.5-VL-3B-Instruct/
```
这一步会用onnx模型替换 vision encoder 模块进行推理。

### 4.转换模型（ONNX -> Axera）

使用模型转换工具 `Pulsar2` 将 ONNX 模型转换成适用于 Axera 的 NPU 运行的模型文件格式 `.axmodel`，通常情况下需要经过以下两个步骤：

- 生成适用于该模型的 PTQ 量化校准数据集
- 使用 `Pulsar2 build` 命令集进行模型转换（PTQ 量化、编译），更详细的使用说明请参考 [AXera Pulsar2 工具链指导手册](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)

1). 生成量化数据集  
这里将图片按照patch编排后，重新保存为图片形式，和onnx模型的输入一致  
```
python get_calib.py
cd calib
tar -cvf hidden_states.tar hidden_states.npy
```

2). 模型转换

* 修改配置文件
 
检查`config.json` 中 `calibration_dataset` 字段，将该字段配置的路径改为上一步下载的量化数据集存放路径  

* Pulsar2 build

参考命令如下：

```
pulsar2 build --input Qwen2.5-VL-3B-Instruct_vision.onnx --config config_video.json --output_dir build-output --output_name Qwen2.5-VL-3B-Instruct_vision.axmodel --target_hardware AX650 --compiler.check 0
```
编译完成后将文件`build-output/Qwen2.5-VL-3B-Instruct_vision.axmodel` 放到 `../Qwen/Qwen2.5-VL-3B-Instruct-AX650/`

## 二、转换 Language Model  

### 1. 转换Language Model  
执行命令
```
pulsar2 llm_build --input_path ../Qwen/Qwen2.5-VL-3B-Instruct/ --output_path ../Qwen/Qwen2.5-VL-3B-Instruct-AX650/ --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 512 --parallel 32 --chip AX650
```
其中 `prefill_len` 的长度就是 `prefill`阶段的最大token数，请根据实际情况设置这个值。

### 2. 从 Language model 中提取 token embeddings  
clone 这个仓库下的工具 https://github.com/AXERA-TECH/ax-llm-build.git   
执行命令  
```
chmod +x ./tools/fp32_to_bf16
chmod +x ./tools/embed_process.sh
./tools/embed_process.sh ../Qwen/Qwen2.5-VL-3B-Instruct/ ../Qwen/Qwen2.5-VL-3B-Instruct-AX650/
```
### 3. 拷贝配置文件
```
cp ../Qwen/Qwen2.5-VL-3B-Instruct/*.json ../Qwen/Qwen2.5-VL-3B-Instruct-AX650/
```

至此，整个模型转换完毕。将../Qwen/Qwen2.5-VL-3B-Instruct-AX650/ 上传到爱芯的设备上准备运行。    
export CUDA_VISIBLE_DEVICES=7

set -e 

CKPT=../../Qwen2.5-VL-7B-Instruct/
python run_nchw.py $CKPT
python export.py $CKPT  image "Qwen2.5-VL-7B-Instruct_vision.onnx"
python test_onnx.py $CKPT "Qwen2.5-VL-7B-Instruct_vision.onnx"
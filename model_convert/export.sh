export CUDA_VISIBLE_DEVICES=7

set -e 

CKPT=../Qwen/Qwen2.5-VL-3B-Instruct/
python run.py $CKPT
python export.py $CKPT
python test_onnx.py $CKPT

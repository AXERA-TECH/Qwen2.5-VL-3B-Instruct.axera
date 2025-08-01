export CUDA_VISIBLE_DEVICES=7

set -e 

CKPT=../../Qwen2.5-VL-7B-Instruct/
python run_video_by_sec.py $CKPT
python export.py  $CKPT video "Qwen2.5-VL-7B-Instruct_vision_video.onnx"
python test_onnx_video_by_sec.py $CKPT "Qwen2.5-VL-7B-Instruct_vision_video.onnx"
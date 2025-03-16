CKPT_DIR=~/AI-support/Qwen/Qwen2.5-VL-3B-Instruct/
OUT_DIR=/data/tmp/yongqiang/nfs/lhj/Qwen2.5-VL-3B-Instruct-AX650-video-prefill_512-back/

pulsar2 llm_build --input_path ${CKPT_DIR} --output_path ${OUT_DIR} --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 512 --parallel 36 --chip AX650

bash tools/embed_process.sh  ${CKPT_DIR}  ${OUT_DIR}

cp ${CKPT_DIR}*.json ${OUT_DIR}
set -e 

pulsar2 llm_build --input_path ~/AI-support/Qwen/Qwen2.5-VL-3B-Instruct/ \
                --output_path ~/AI-support/Qwen/Qwen2.5-VL-3B-Instruct-AX650-chunk_prefill_512/ \
                --kv_cache_len 1023 \
                --hidden_state_type bf16 \
                --prefill_len 128 \
                --last_kv_cache_len 128 \
                --last_kv_cache_len 256 \
                --last_kv_cache_len 384 \
                --last_kv_cache_len 512 \
                --parallel 32 --chip AX650

./tools/embed_process.sh ~/AI-support/Qwen/Qwen2.5-VL-3B-Instruct/ ~/AI-support/Qwen/Qwen2.5-VL-3B-Instruct-AX650-chunk_prefill_512/
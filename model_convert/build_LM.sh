set -e 

pulsar2 llm_build --input_path ../..//Qwen2.5-VL-7B-Instruct/ \
                --output_path ../..//Qwen2.5-VL-7B-Instruct-AX650-chunk_prefill_1280/ \
                --kv_cache_len 2047 \
                --hidden_state_type bf16 \
                --prefill_len 128 \
                --last_kv_cache_len 128 \
                --last_kv_cache_len 256 \
                --last_kv_cache_len 384 \
                --last_kv_cache_len 512 \
                --last_kv_cache_len 640 \
                --last_kv_cache_len 768 \
                --last_kv_cache_len 896 \
                --last_kv_cache_len 1024 \
                --last_kv_cache_len 1152 \
                --last_kv_cache_len 1280 \
                --chip AX650

./tools/embed_process.sh ../..//Qwen2.5-VL-7B-Instruct/ ../..//Qwen2.5-VL-7B-Instruct-AX650-chunk_prefill_1280/
#!/bin/bash

input="$1"
output="$2"

python tools/extract_embed.py --input_path $input --output_path $output
python tools/embed-process.py --input $output/model.embed_tokens.weight.npy --output $output/model.embed_tokens.weight.float32.bin
./tools/fp32_to_bf16 $output/model.embed_tokens.weight.float32.bin $output/model.embed_tokens.weight.bfloat16.bin

import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input",
                    type=str,
                    required=True,
                    default="llama2.model.embed_tokens.weight.npy")
parser.add_argument("--output",
                    type=str,
                    required=True,
                    default="llama2.model.embed_tokens.weight.float32.bin")
args = parser.parse_args()

input_data = np.load(args.input, allow_pickle=True)
print(input_data.shape)

with open(args.output, "wb") as f:
    f.write(input_data.astype(np.float32).tobytes())

import argparse
import pathlib

import numpy as np
import safetensors
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, help="input model path", required=True)
    parser.add_argument("--output_path", type=str, help="embed output path", required=False)

    args = parser.parse_args()

    input_path = pathlib.Path(args.input_path)
    output_path = pathlib.Path(args.output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    bin_file = input_path / "pytorch_model.bin"
    if bin_file.exists():
        torch_bin = torch.load(bin_file.as_posix(), map_location="cpu", mmap=True)
        if "model.embed_tokens.weight" in torch_bin:
            print(f"find model.embed_tokens.weight in pytorch_model.bin")
            key = "model.embed_tokens.weight"
        elif "llm.model.embed_tokens.weight" in torch_bin:
            print(f"find llm.model.embed_tokens.weight in pytorch_model.bin")
            key = "llm.model.embed_tokens.weight"
        elif "language_model.model.embed_tokens.weight" in torch_bin:
            print(f"find anguage_model.model.embed_tokens.weight in pytorch_model.bin")
            key = "language_model.model.embed_tokens.weight"
        embeds_pt = torch_bin[key]
        embeds_np = embeds_pt.detach().to(torch.float32).numpy()
        np.save(output_path / f"model.embed_tokens.weight.npy", embeds_np)
    else:
        sf_files = input_path.glob("*.safetensors")
        for file in sf_files:
            find = False
            with safetensors.safe_open(file, "pt", device="cpu") as f:
                for k in f.keys():
                    if k in ["model.embed_tokens.weight",
                                "llm.model.embed_tokens.weight",
                                "language_model.model.embed_tokens.weight",
                                "embed_tokens.weight", 
                                "model.text_model.embed_tokens.weight"]:
                        #
                        print(f"find {k} in {file.as_posix()}")
                        res = f.get_tensor(k).to(torch.float32).numpy()
                        np.save(output_path / f"model.embed_tokens.weight.npy", res)
                        find = True
                        break
            if find:
                break

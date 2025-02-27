from transformers import AutoTokenizer, AutoConfig
import numpy as np
from ml_dtypes import bfloat16
from axengine import InferenceSession
from PIL import Image
from torchvision import transforms
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import torch
from transformers import  AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import onnxruntime
import gc


def post_process(data, topk=1, topp=0.9, temperature=0.6):
    def top_p(l: np.ndarray, p: float) -> np.ndarray:
        index = np.argsort(l)
        res = l.copy()
        sum_p = 0
        for i in index[::-1]:
            if sum_p >= p:
                res[i] = 0
            sum_p += res[i]
        return res / sum_p

    def softmax(l: np.ndarray) -> np.ndarray:
        l_max = l - l.max()
        l_exp = np.exp(l_max)
        res = l_exp / np.sum(l_exp)
        return res.astype(np.float64)

    r = data.astype(np.float32)
    r = r.flatten()
    # topk
    candidate_index = np.argpartition(r, -topk)[-topk:]
    candidate_value = r[candidate_index]
    # temperature
    candidate_value /= temperature
    # softmax
    candidate_soft = softmax(candidate_value)
    # topp
    candidate_soft = top_p(candidate_soft, topp)
    candidate_soft = candidate_soft.astype(np.float64) / candidate_soft.sum()
    pos = np.random.multinomial(1, candidate_soft).argmax()
    next_token = candidate_index[pos]
    return next_token, candidate_index, candidate_soft

POSTION_IDS = torch.tensor([[[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 31,
          32, 33, 34, 35, 36, 37, 38, 39, 40]],

        [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 15,
          15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16,
          16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17,
          17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18,
          18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19,
          19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20,
          20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21,
          21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 22,
          22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23,
          23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
          24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
          25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
          26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
          27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
          28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29, 29,
          30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31,
          32, 33, 34, 35, 36, 37, 38, 39, 40]],

        [[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 15, 16, 17,
          18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 15, 16, 17, 18,
          19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 15, 16, 17, 18, 19,
          20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 15, 16, 17, 18, 19, 20,
          21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 15, 16, 17, 18, 19, 20, 21,
          22, 23, 24, 25, 26, 27, 28, 29, 30, 15, 16, 17, 18, 19, 20, 21, 22,
          23, 24, 25, 26, 27, 28, 29, 30, 15, 16, 17, 18, 19, 20, 21, 22, 23,
          24, 25, 26, 27, 28, 29, 30, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
          25, 26, 27, 28, 29, 30, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
          26, 27, 28, 29, 30, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
          27, 28, 29, 30, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
          28, 29, 30, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
          29, 30, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
          30, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
          15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
          32, 33, 34, 35, 36, 37, 38, 39, 40]]])


if __name__ == "__main__":
    
    checkpoint_dir="../Qwen/Qwen2.5-VL-3B-Instruct-AX650-mrope/"
    cfg = AutoConfig.from_pretrained(
        checkpoint_dir, trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, trust_remote_code=True
    )
        

    processor = AutoProcessor.from_pretrained(checkpoint_dir) 
    messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                            # "image": "demo.jpg"
                            "image": "demo1.jpg"
                        },
                        {"type": "text", "text": "Describe this image."},
                    ],
                }
            ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    pixel_values = inputs['pixel_values']
    # extract img feature by vit
    vit_session = InferenceSession.load_from_model('../Qwen/Qwen2.5-VL-3B-Instruct-AX650-mrope/Qwen2.5-VL-3B-Instruct_vision.axmodel')
    vit_output = vit_session.run({"hidden_states": pixel_values.numpy()})[0]  # (1, 256, 2048)

    vit_output = vit_output[None,:,:]
    print("vit feature extract done!")
    
    del vit_session
    gc.collect()

    token_ids = inputs['input_ids'].squeeze().numpy().tolist()

    image_start_index = np.where(np.array(token_ids) == 151652)[0].tolist()[0]
    image_insert_index = image_start_index + 1
    embeds = np.load("../Qwen/Qwen2.5-VL-3B-Instruct-AX650-mrope/model.embed_tokens.weight.npy")
    prefill_data = np.take(embeds, token_ids, axis=0)
    prefill_data = prefill_data.astype(bfloat16)
    
    prefill_data[ image_insert_index : image_insert_index + 256] = vit_output[0, :, :]
    token_len = len(token_ids)


    lastN = 1023
    kv_dim = cfg.hidden_size // cfg.num_attention_heads * cfg.num_key_value_heads
    k_caches = [
        np.zeros((1, lastN, kv_dim), dtype=bfloat16)
        for _ in range(cfg.num_hidden_layers)
    ]
    v_caches = [
        np.zeros((1, lastN, kv_dim), dtype=bfloat16)
        for _ in range(cfg.num_hidden_layers)
    ]

    prefill_decoder_sessins = []
    for i in range(cfg.num_hidden_layers):
        session = InferenceSession.load_from_model(
            f"../Qwen/Qwen2.5-VL-3B-Instruct-AX650-mrope/qwen2_5_vl_p320_l{i}_together.axmodel"
        )
        prefill_decoder_sessins.append(session)
    post_process_session = InferenceSession.load_from_model(
        "../Qwen/Qwen2.5-VL-3B-Instruct-AX650-mrope/qwen2_5_vl_post.axmodel"
    )
    print("model load done!")

    """
        prefill
    """
    prefill_len = 320
    for i in range(cfg.num_hidden_layers):
        prefill_decoder_sessins[i].set_runtime_context(group_id=1)

    if prefill_len > 0:
        indices = np.zeros((3, prefill_len), dtype=np.uint32)

        indices[:, 0:token_len] = POSTION_IDS.squeeze(1).numpy().astype(np.uint32)

        mask = np.zeros((1, prefill_len, prefill_len)) - 65536
        data = np.zeros((1, prefill_len, cfg.hidden_size)).astype(bfloat16)
        
        data[:, 0:token_len] = prefill_data
        for i, t in enumerate(token_ids):
            mask[:, i, : i + 1] = 0
        mask = mask.astype(bfloat16)
        for i in range(cfg.num_hidden_layers):
            input_feed = {
                "K_cache": np.zeros((1, 1, cfg.hidden_size), dtype=bfloat16),
                "V_cache": np.zeros((1, 1, cfg.hidden_size), dtype=bfloat16),
                "indices": indices,
                "input": data,
                "mask": mask,
            }
            outputs = prefill_decoder_sessins[i].run(input_feed)
            k_caches[i][:, :token_len, :] = outputs[0][:, :token_len, :]
            v_caches[i][:, :token_len, :] = outputs[1][:, :token_len, :]
            data = outputs[2][:, :token_len, :]

    post_out = post_process_session.run({"input": data[:, token_len - 1, :]})[0]
    next_token, posssible_tokens, possible_soft = post_process(post_out, topk=1)
    posibles = [tokenizer.decode([t]) for t in posssible_tokens]
    posible_soft = [str((t, s)) for t, s in zip(posibles, possible_soft)]
    token_ids.append(next_token)
    print("prefill done!")
  
    # set to decoder
    for i in range(cfg.num_hidden_layers):
        prefill_decoder_sessins[i].set_runtime_context(group_id=0)

    # lastN = np.max(indices)
    start_ids = np.max(indices) + 1
    mask = np.zeros((1, 1, lastN + 1), dtype=np.float32).astype(bfloat16)
    mask[:, :, :lastN] -= 65536
    mask[:, :, :token_len] = 0
    for start_indice in range(lastN + 1):
        if prefill_len > 0 and start_indice < token_len:
            continue
        next_token = token_ids[start_indice]
        indices = np.array([start_ids], np.uint32).reshape((1, 1))
        start_ids += 1
        data = embeds[next_token, :].reshape((1, 1, cfg.hidden_size)).astype(bfloat16)

        for i in range(cfg.num_hidden_layers):
            input_feed = {
                "K_cache": k_caches[i],
                "V_cache": v_caches[i],
                "indices": indices,
                "input": data,
                "mask": mask,
            }
            outputs = prefill_decoder_sessins[i].run(input_feed)
            k_caches[i][:, start_indice, :] = outputs[0][:, :, :]
            v_caches[i][:, start_indice, :] = outputs[1][:, :, :]
            data = outputs[2]
        mask[..., start_indice] = 0
        if start_indice < token_len - 1:
            pass
        else:
            post_out = post_process_session.run({"input": data})[0]
            next_token, posssible_tokens, possible_soft = post_process(post_out)
            token_ids.append(next_token)
        if next_token == tokenizer.eos_token_id:
            # print("hit eos!")
            break
    print(tokenizer.decode(token_ids[token_len:]))
    
    
    
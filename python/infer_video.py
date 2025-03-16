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
from glob import glob
from utils import get_rope_index
from transformers.image_utils import PILImageResampling
from preprocess import Qwen2VLImageProcessorExport

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



if __name__ == "__main__":

    prefill_len = 512
    
    checkpoint_dir=f"../Qwen2.5-VL-3B-Instruct-AX650-video-prefill_512/"
    cfg = AutoConfig.from_pretrained(
        checkpoint_dir, trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, trust_remote_code=True
    )
        
    processor = AutoProcessor.from_pretrained(checkpoint_dir) 
    paths = sorted(glob("demo_cv308/*.jpg"))
    print(paths)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": paths,
                    "max_pixels": 308 * 308,
                    "fps": 1.0,
                },
                {"type": "text", "text": "描述一下这个视频的内容"},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("text",text)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    position_ids,_ = get_rope_index(cfg, inputs["input_ids"], video_grid_thw=inputs['video_grid_thw'], second_per_grid_ts=inputs['second_per_grid_ts'])

    # pixel_values = inputs['pixel_values_videos']
    # print("pixel_values",pixel_values.shape)
    # extract img feature by vit
    vit_session = InferenceSession.load_from_model(f'{checkpoint_dir}/Qwen2.5-VL-3B-Instruct_vision_nhwc.axmodel')

    t = inputs['video_grid_thw'][0,0]

    images = []
    for p in paths:
        img = Image.open(p)
        images.append(img)

    img_processor = Qwen2VLImageProcessorExport(max_pixels=308*308, patch_size=14, temporal_patch_size=2, merge_size=2)
    pixel_values, grid_thw = img_processor._preprocess(images, do_resize=True, resample=PILImageResampling.BICUBIC, 
                                        do_rescale=False, do_normalize=False, 
                                        do_convert_rgb=True)

    # seq_len, dim = pixel_values.shape
    # ht = pixel_values.reshape(t, seq_len//t, dim)
    print("pixel_values.shape",pixel_values.shape)
    t, seq_len,_,_ = pixel_values.shape
    ht = pixel_values
    vit_output = []
    for i in range(t):
        out = vit_session.run({"hidden_states": ht[i]})[0]  # (1, 576, 1176)
        vit_output.append(out.astype(bfloat16))
    
    del vit_session
    gc.collect()

    vit_output = np.concatenate(vit_output, axis=0)
    vit_output = vit_output[None,:,:]
    
    print("vit feature extract done!")

    token_ids = inputs['input_ids'].squeeze().numpy().tolist()

    image_start_index = np.where(np.array(token_ids) == 151652)[0].tolist()[0]
    image_insert_index = image_start_index + 1
    embeds = np.load(f"{checkpoint_dir}/model.embed_tokens.weight.npy")
    prefill_data = np.take(embeds, token_ids, axis=0)
    prefill_data = prefill_data.astype(bfloat16)
    
    prefill_data[ image_insert_index : image_insert_index + vit_output.shape[1]] = vit_output[0, :, :]
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
            f"{checkpoint_dir}/qwen2_5_vl_p{prefill_len}_l{i}_together.axmodel"
        )
        prefill_decoder_sessins.append(session)
    post_process_session = InferenceSession.load_from_model(
        f"{checkpoint_dir}/qwen2_5_vl_post.axmodel"
        # "../Qwen2.5-VL-3B-Instruct-AX650-video-prefill_512/qwen2_5_vl_post.axmodel"
    )
    print("model load done!")

    """
        prefill
    """
    
    for i in range(cfg.num_hidden_layers):
        prefill_decoder_sessins[i].set_runtime_context(group_id=1)

    if prefill_len > 0:
        indices = np.zeros((3, prefill_len), dtype=np.uint32)

        indices[:, 0:token_len] = position_ids.squeeze(1).numpy().astype(np.uint32)

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

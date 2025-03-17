import torch
from transformers import  AutoTokenizer, AutoProcessor, AutoConfig
from modeling_qwen2_5_vl_export import Qwen2_5_VLForConditionalGenerationInfer
from qwen_vl_utils import process_vision_info
import sys 
from PIL import Image
from utils import get_rope_index
from glob import glob
import numpy as np 

checkpoint_dir = sys.argv[1] if len(sys.argv)>=2 else "../../Qwen/Qwen2.5-VL-3B-Instruct/"
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGenerationInfer.from_pretrained(
    checkpoint_dir, torch_dtype=torch.float32, device_map="cuda"
)

paths = sorted(glob("../demo/*.jpg"))
print(paths)

text = "描述这个视频."
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
            {"type": "text", "text": text},
        ],
    }
]



#In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
# Preparation for inference
cfg = AutoConfig.from_pretrained(
        checkpoint_dir, trust_remote_code=True
    )
processor = AutoProcessor.from_pretrained(checkpoint_dir) 
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)
inputs = inputs.to("cuda")
# position_ids,_ = get_rope_index(cfg, inputs["input_ids"], video_grid_thw=inputs['video_grid_thw'], second_per_grid_ts=inputs['second_per_grid_ts'])

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
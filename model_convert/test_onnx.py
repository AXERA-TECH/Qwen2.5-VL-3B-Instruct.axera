import torch
from transformers import  AutoTokenizer, AutoProcessor
from modeling_qwen2_5_vl_export import Qwen2_5_VLForConditionalGenerationExport
from qwen_vl_utils import process_vision_info
import sys

checkpoint_dir = sys.argv[1] if len(sys.argv)>=2 else "../../Qwen/Qwen2.5-VL-3B-Instruct/"
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGenerationExport.from_pretrained(
    checkpoint_dir, torch_dtype=torch.float32, device_map="cuda"
)

if len(sys.argv)>2 and sys.argv[2]=="two_parts":
    model.visual.forward = model.visual.forward_onnx_two_parts
else:
    model.visual.forward = model.visual.forward_onnx
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained(checkpoint_dir)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                "image": "../assets/demo.jpg"
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

inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
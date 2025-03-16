import torch
from transformers import  AutoTokenizer, AutoProcessor
from modeling_qwen2_5_vl_export import Qwen2_5_VLForConditionalGenerationExport
from qwen_vl_utils import process_vision_info
import sys
from glob import glob
from PIL import Image
from transformers.image_utils import PILImageResampling
from preprocess import Qwen2VLImageProcessorExport

checkpoint_dir = sys.argv[1] if len(sys.argv)>=2 else "../../Qwen/Qwen2.5-VL-3B-Instruct/"
# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGenerationExport.from_pretrained(
    checkpoint_dir, torch_dtype=torch.float32, device_map="cuda"
)
# model.visual.forward = model.visual.forward_onnx_by_second

model.visual.forward = model.visual.forward_onnx_by_second_1


paths = sorted(glob("../demo/*.jpg"))
print(paths)
paths=paths

images = []
for p in paths:
    img = Image.open(p)
    images.append(img)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": images,
                "max_pixels": 308 * 308,
                "fps": 1.0,
            },
            {"type": "text", "text": "描述这个视频."},
        ],
    }
]

img_processor = Qwen2VLImageProcessorExport(max_pixels=308*308, patch_size=14, temporal_patch_size=2, merge_size=2)

image_mean = [
    0.48145466,
    0.4578275,
    0.40821073
  ]

image_std =  [
    0.26862954,
    0.26130258,
    0.27577711
  ]
# pixel_values, grid_thw = img_processor._preprocess(images, do_resize=True, resample=PILImageResampling.BICUBIC, 
#                                     do_rescale=True, rescale_factor=1/255, do_normalize=True, 
#                                     image_mean=image_mean, image_std=image_std,do_convert_rgb=True)
pixel_values, grid_thw = img_processor._preprocess(images, do_resize=True, resample=PILImageResampling.BICUBIC, 
                                        do_rescale=False, do_normalize=False, 
                                        do_convert_rgb=True)

pixel_values = torch.from_numpy(pixel_values).to("cuda")
mean = torch.tensor(image_mean,dtype=torch.float32).reshape([1,1,1,3])*255
mean = mean.to("cuda")
std = torch.tensor(image_std,dtype=torch.float32).reshape([1,1,1,3])*255
std = std.to("cuda")
pixel_values = (pixel_values-mean)/std
pixel_values = pixel_values.permute(0,3,1,2)
# Preparation for inference
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


inputs['pixel_values_videos'] = pixel_values

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
# from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from preprocess import Qwen2VLImageProcessorExport
from transformers.image_utils import PILImageResampling
from glob import glob 
import numpy as np
import random
from PIL import Image
import cv2
if __name__=="__main__":
    paths = sorted(glob("demo/*"))[75:90][::2]

    # paths = random.sample(paths, 64)
    # paths = sorted(paths)
    print(paths)

    images = []
    for p in paths:
        img = Image.open(p)
        img = img.crop([150, 540-512, 150+512, 540])
        images.append(img)

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
    print("pixel_values_videos", pixel_values.shape)

    t = len(images)//2

    # thw, dim = pixel_values.shape
    # pixel_values = pixel_values.reshape(t, thw//t, dim)

    t = pixel_values.shape[0]
    for i in range(t):
        print("hi shape",pixel_values[i].shape)
        # np.save(f"calib/h{i}.npy", pixel_values[i])
        cv2.imwrite(f"calib/h{i}.jpg", pixel_values[i].astype(np.uint8))
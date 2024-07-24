# -*- coding: utf-8 -*-
# Inference with Video-LLaVa
## Set-up environment
"""
pip install --upgrade -q accelerate bitsandbytes
pip install git+https://github.com/huggingface/transformers.git
pip install -q av
"""

import torch
from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "LanguageBind/Video-LLaVA-7B-hf"

processor = VideoLlavaProcessor.from_pretrained(model_id)
model = VideoLlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

import av
import numpy as np

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

from huggingface_hub import hf_hub_download

# Download video from the hub
video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
container = av.open(video_path)

# sample uniformly 8 frames from the video
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
clip = read_video_pyav(container, indices)

prompt = "USER: <video>\nWhy is this video funny? ASSISTANT:"

inputs = processor(prompt, videos=clip, return_tensors="pt").to(model.device)
for k,v in inputs.items():
    print(k,v.shape)

generate_kwargs = {"max_new_tokens":100, "do_sample":True, "top_p":0.9, "top_k":2}

output = model.generate(**inputs, **generate_kwargs)
generated_text = processor.batch_decode(output, skip_special_tokens=True)

print(generated_text[0])

import requests
from PIL import Image

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
url = "https://mma.prnewswire.com/media/2237803/Bosch_Power_Tools_Products.jpg?p=publish"
image = Image.open(requests.get(url, stream=True).raw)
image

# This time we will use a special "<image>" token instead of "<video>"
prompt = "USER: <image>\nOf which company the tools belongs to in the image? ASSISTANT:"
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

# Generate
generate_ids = model.generate(**inputs, **generate_kwargs)
generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True)
print(generated_text[0])

prompt = "USER: <image>\nHow many cats are there in the image? ASSISTANT: There are two cats in the image. USER: <video>\nWhy is this video funny? ASSISTANT:"
inputs = processor(text=prompt, images=image, videos=clip, padding=True, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, **generate_kwargs)
generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True)
print(generated_text[0])

from huggingface_hub import hf_hub_download
prompts = ["USER: <image>\nHow many cats are there in the image? ASSISTANT:", "USER: <video>\nWhy is this video funny? ASSISTANT:"]
inputs = processor(text=prompts, images=image, videos=clip, padding=True, return_tensors="pt").to(model.device)

# Generate
generate_ids = model.generate(**inputs, **generate_kwargs)
generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True)
print(generated_text)
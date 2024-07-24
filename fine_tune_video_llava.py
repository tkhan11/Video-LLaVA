# -*- coding: utf-8 -*-
## Fine-tune VIdeo-LLaVa on MMBench dataset
"""
## Prerequisites
Before we start, make sure you have the following:

- Access to a GPU (preferably A100 since videos require high sequence lengths).
- Familiarity with Hugging Face’s Transformers library.
- Pre-install necessary packages by running the below.

!pip install -U -q transformers accelerate bitsandbytes peft dataset
!pip install -q av
!pip install -q lightning

"""
import os
import av
import re
import bisect
import shutil
import numpy as np
from nltk import edit_distance

from transformers import AutoProcessor
from transformers import BitsAndBytesConfig, VideoLlavaForConditionalGeneration
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download
from datasets import load_dataset, concatenate_datasets

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


MAX_LENGTH = 256
MODEL_ID = "LanguageBind/Video-LLaVA-7B-hf"
REPO_ID = "RaushanTurganbay/VideoLLava-demo" # Change to your hf-hub repo

USE_LORA = False
USE_QLORA = True

config2path = {
    "object_interaction": "star/Charades_v1_480",
    "action_sequence": "star/Charades_v1_480",
    "action_prediction": "star/Charades_v1_480",
    "moving_count": "clevrer/video_validation",
    "moving_attribute": "clevrer/video_validation",
    "object_existence": "clevrer/video_validation",
    "moving_direction": "clevrer/video_validation",
    "counterfactual_inference": "clevrer/video_validation",
    "unexpected_action": "FunQA_test/test",
    "episodic_reasoning": "tvqa/frames_fps3_hq",
    "action_antonym": "ssv2_video",
    "scene_transition": "scene_qa/video",
    "fine_grained_pose": "nturgbd",
    "object_shuffle": "perception/videos",
    "state_change": "perception/videos",
    "character_order": "perception/videos",
    "action_localization": "sta/sta_video",
    "fine_grained_action": "Moments_in_Time_Raw/vi",
    "egocentric_navigation": "vlnqa",
}

def read_video_pyav(video_path, start, end):
    """Reads a video for given start-end timestamps interval and uniformly samples 8 frames of it"""
    container = av.open(video_path)
    video = container.streams.get(0)[0]

    av_timestamps = [
        int(packet.pts * video.time_base) for packet in container.demux(video) if packet.pts is not None
    ]

    av_timestamps.sort()
    start_id = bisect.bisect_left(av_timestamps, start)
    end_id = bisect.bisect_left(av_timestamps, end)

    # in case it is a very short video, lets take a longer duration and sample
    if end_id  - start_id < 10:
        end_id += 10
        start_id -= 10

    end_id = min(len(av_timestamps) - 1, end_id)
    start_id = max(1, start_id)

    # We sample 8 frames for tuning following the original paper
    # But we can increase the number of frames for longer videos and check out if it helps performance
    # Change the below "8" to any number of frames you want, and note that more frames -> more computational resources needed
    indices = np.linspace(start_id, end_id, 8).astype(int)

    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_id:
            break
        if i >= start_id and i in indices:
            frames.append(frame)
    assert len(frames) == 8, f"Got {len(frames)} frames but should be 8. Check the indices: {indices};, start_id: {start_id}, end_id: {end_id}. Len of video is {len(av_timestamps)} frames."
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def collate_read_video(example, path):
    # Some datasets have a start-end interval, so we try to get it if exists. Otherwise just set a very large end timestamp
    clip = read_video_pyav(f'{path}/{example["video"]}', example.get("start", 1), example.get("end", 1e+10))
    example["clip"] = clip
    return example

# Download the videos from datasets repo and unzip. Make sure you have enough free space before downloading and unzipping
videos = snapshot_download(repo_id="OpenGVLab/MVBench", allow_patterns="*", repo_type="dataset")
for zip_file in os.listdir(f"{videos}/video"):
    if zip_file.endswith(".zip"):
        shutil.unpack_archive(f"{videos}/video/{zip_file}", f"{videos}/videos_unzipped/")

# Load each config and save in a mapping
config2ds = {}
for config, path in config2path.items():
    ds = load_dataset("OpenGVLab/MVBench", config, split="train")
    ds = ds.map(collate_read_video, batched=False, fn_kwargs={"path": f"{videos}/videos_unzipped/{path}"})
    config2ds[config] = ds

processor = AutoProcessor.from_pretrained(MODEL_ID)
processor.tokenizer.padding_side = "right" # during training, one always uses padding on the right


class VideoLlavaDataset(Dataset):
    """
    PyTorch Dataset for VideoLlavaDataset. This class takes a HuggingFace Dataset as input.
    """

    def __init__(
        self,
        dataset: str,
    ):
        super().__init__()
        self.dataset = dataset
        self.id2choice = {0: "A", 1: "B", 2: "C", 3: "D"}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        # cast to np because ds.map() casted everything to list, and the processor does not work list format
        clip = np.array(sample["clip"])

        question, candidates = sample["question"], sample["candidates"]
        answer = candidates.index(sample["answer"])
        answer = self.id2choice[answer]

        mult_choice = ""
        for i, choice in enumerate(candidates):
            mult_choice += f"{self.id2choice[i]}. {choice}; "

        # Prepare a prompt template, can be changed depeding on the dataset and use-cases
        prompt = f"USER: <video>\nAnswer the following multiple choice question based on the video. " \
                f"Question: {question}\n {mult_choice}\n ASSISTANT: Answer: {answer}"

        return prompt, clip

def train_collate_fn(examples):
    videos = []
    texts = []
    texts, videos = list(zip(*examples))

    batch = processor(text=texts, videos=videos, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    labels = batch["input_ids"].clone()

    # We don't want to compute loss for pad tokens, lets mask with -100. Some methods also mask the prompt, calculating loss only on the answers/captions/etc
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values_videos = batch["pixel_values_videos"]
    labels = batch["labels"]

    return input_ids, attention_mask, pixel_values_videos, labels


def eval_collate_fn(examples):
    # We only feed the prompt to the model
    # Make sure to separate prompt from answers/captions/etc depending on your own task and dataset
    # Otherwise your model will peek into the ground truth
    videos = []
    texts = []
    texts, videos = list(zip(*examples))
    texts = [text[:-2] for text in texts]  # Get text without answers, so the model has to generate the answers itself during eval

    batch = processor(text=texts, videos=videos, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values_videos = batch["pixel_values_videos"]
    answer_choice = [texts[-1] for text in texts] # Save answer's (one last letter choice) to calc accuracy later

    return input_ids, attention_mask, pixel_values_videos, answer_choice

datasets_combined = concatenate_datasets(list(config2ds.values()))
datasets_combined = datasets_combined.shuffle(seed=42)
dataset = datasets_combined.train_test_split(test_size=0.2)

dataset

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML


example = dataset['train'][0]
clip = example["clip"]

# np array with shape (frames, height, width, channels)
video = np.array(clip)

fig = plt.figure()
im = plt.imshow(video[0,:,:,:])

plt.close() # this is required to not display the generated image

def init():
    im.set_data(video[0,:,:,:])

def animate(i):
    im.set_data(video[i,:,:,:])
    return im

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0],
                               interval=100)
HTML(anim.to_html5_video())

# See what was the questions and ground truth answer
example["question"], example["candidates"], example["answer"]

"""And now we wrap it in the Pytorch Datasets class and print one example as sanity check."""

train_dataset = VideoLlavaDataset(dataset["train"])
eval_dataset = VideoLlavaDataset(dataset["test"])

prompt, clip = train_dataset[0]

# Seems like we're good to go
prompt

## Load model
# Three options for training, from the lowest precision training to the highest precision training:
# QLoRA: model uses 4-bit quantization, which helps in reducing memory usage while maintaining performance.
# Standard LoRA:  model is loaded with standard LoRA adaptations.
# Full Fine-Tuning: no memory optimization are done. In that case Flash Attention is used to speed up training, if hardware supports it.

if USE_QLORA or USE_LORA:
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",
    )
else:
    # for full fine-tuning, we can speed up the model using Flash Attention
    # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
        device_map="auto",
    )

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=find_all_linear_names(model),
    init_lora_weights="gaussian",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

model

class VideoLlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model

        self.batch_size = config.get("batch_size")

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, pixel_values_videos, labels = batch

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            labels=labels
        )
        loss = outputs.loss

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

        input_ids, attention_mask, pixel_values_videos, answers = batch

        # autoregressively generate token IDs
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            max_new_tokens=MAX_LENGTH,
            do_sample=False,
        )
        # turn them back into text, chopping of the prompt
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)

        correct = 0
        for pred, answer in zip(predictions, answers):
            correct += (pred.strip().lower() == answer.lower())
        self.log("val_accuracy", correct / len(answers))

        return correct

    def configure_optimizers(self):
        # you could also add a learning rate scheduler if you want
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))

        return optimizer

    def train_dataloader(self):
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(eval_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)

config = {"max_epochs": 2,
          # "val_check_interval": 0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8,
          "lr": 1e-4,
          "batch_size": 1,
          "num_nodes": 1,
          "warmup_steps": 50,
}

model_module = VideoLlavaModelPLModule(config, processor, model)
early_stop_callback = EarlyStopping(monitor="val_accuracy", patience=3, verbose=False, mode="min")

from huggingface_hub import HfApi

api = HfApi()

class PushToHubCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        print(f"Pushing model to the hub, epoch {trainer.current_epoch}")
        pl_module.model.push_to_hub(REPO_ID,
                                    commit_message=f"Training in progress, epoch {trainer.current_epoch}")

    def on_train_end(self, trainer, pl_module):
        print(f"Pushing model to the hub after training")
        pl_module.processor.push_to_hub(REPO_ID,
                                    commit_message=f"Training done")
        pl_module.model.push_to_hub(REPO_ID,
                                    commit_message=f"Training done")

early_stop_callback = EarlyStopping(monitor="val_edit_distance", patience=3, verbose=False, mode="min")

trainer = L.Trainer(
        default_root_dir="/raid/.cache/huggingface/video_llava_demo",
        accelerator="gpu",
        devices=[0],
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        limit_val_batches=5,
        num_sanity_val_steps=1,
        callbacks=[early_stop_callback, PushToHubCallback()],
)

trainer.fit(model_module)

from transformers import AutoProcessor, BitsAndBytesConfig, VideoLlavaForConditionalGeneration
import torch

processor = AutoProcessor.from_pretrained(MODEL_ID)

# Define quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the base model with adapters on top
model = VideoLlavaForConditionalGeneration.from_pretrained(
    REPO_ID,
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    device_map="auto",
)

"""Now we're ready to perform inference. We'll take a one example from the validation set here and plot 8 frames to see what is happening in the video."""

from matplotlib import pyplot as plt
from PIL import Image

prompt, clip = eval_dataset[1]
fig, axarr = plt.subplots(2, 4, figsize = (10, 10))
fig.tight_layout()

for i in range(2):
    for j in range(4):
        curr_frame = Image.fromarray(np.uint8(clip[i + j]))
        axarr[i, j].imshow(curr_frame)
        axarr[i, j].get_xaxis().set_visible(False)
        axarr[i, j].get_yaxis().set_visible(False)
        axarr[i, j].set_aspect('equal')

plt.subplots_adjust(wspace=None, hspace=None)
plt.axis('off')
plt.show()

answer = prompt[-1]
prompt = prompt[:-2]

inputs = processor(text=prompt, videos=clip, return_tensors="pt").to(model.device)
for k,v in inputs.items():
    print(k,v.shape)

# Generate token IDs
generated_ids = model.generate(**inputs, max_new_tokens=MAX_LENGTH)
# Decode back into text
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print(generated_texts)
print(answer)


#!/usr/bin/env python

# Run this before you deploy it on replicate
import torch
from diffusers.utils import load_image
from diffusers import DiffusionPipeline, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
import os
import sys
import urllib

# append project directory to path so predict.py can be imported
sys.path.append('.')
from predict import MODEL,CONTROLNET,MODEL_ID, CONTROLNET_CACHE, MODEL_ID_CACHE, MODEL_CACHE

# Make cache folders
if not os.path.exists(CONTROLNET_CACHE):
    os.makedirs(CONTROLNET_CACHE)

if not os.path.exists(MODEL_ID_CACHE):
    os.makedirs(MODEL_ID_CACHE)

if not os.path.exists(MODEL_CACHE):
    os.makedirs(MODEL_CACHE)


model = OpenposeDetector.from_pretrained(
    MODEL,
    cache_dir=MODEL_CACHE,
)

controlnet = ControlNetModel.from_pretrained(
    CONTROLNET,
    torch_dtype=torch.float16,
    cache_dir=CONTROLNET_CACHE,
)

controlnet.save_pretrained(CONTROLNET_CACHE)

pipe = DiffusionPipeline.from_pretrained(
     MODEL_ID,
     torch_dtype=torch.float16,
     cache_dir=MODEL_ID_CACHE,
    )
pipe.save_pretrained(MODEL_ID_CACHE, safe_serialization=True)
# if not os.path.exists(MODEL_ID_CACHE + "/realitycheckXL_alpha11.safetensors"):
#     print("Downloading model realitycheckXL_alpha11 ")
#     urllib.request.urlretrieve("https://stableai-space.fra1.digitaloceanspaces.com/models/sdxl_models/realitycheckXL_alpha11.safetensors", MODEL_ID_CACHE + "/realitycheckXL_alpha11.safetensors")

#check if the file exists
if not os.path.exists("selfie_multiclass_256x256.tflite"):
    # Download the file from `url` and save it locally under `file_name`:
    print("Downloading selfie_multiclass_256x256.tflite")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite", "selfie_multiclass_256x256.tflite")
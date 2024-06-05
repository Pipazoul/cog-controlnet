# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from PIL import Image

import insightface
import onnxruntime
from insightface.app import FaceAnalysis
import cv2
import gfpgan

import base64
import cv2
import numpy as np
import random
from cog import BasePredictor, Input, Path
import tempfile
import time

MODEL = "lllyasviel/ControlNet"
CONTROLNET = "fusing/stable-diffusion-v1-5-controlnet-openpose"
MODEL_ID = "Lykon/AbsoluteReality"

CONTROLNET_CACHE = "control-cache"
MODEL_CACHE = "model-cache"
MODEL_ID_CACHE = "model-id-cache"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        # nvidia-smi
        print("torch.cuda.is_available()", torch.cuda.is_available())
        print("torch.cuda.device_count()", torch.cuda.device_count())
        print("torch.cuda.current_device()", torch.cuda.current_device())
        print("torch.cuda.get_device_name(0)", torch.cuda.get_device_name(0))

        self.model = OpenposeDetector.from_pretrained(MODEL, cache_dir=MODEL_CACHE)
        self.controlnet = ControlNetModel.from_pretrained(
            CONTROLNET,
            cache_dir=CONTROLNET_CACHE,
            torch_dtype=torch.float16
        )
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_ID_CACHE,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
        )
        self.pipe.to("cuda")
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()


        self.face_swapper = insightface.model_zoo.get_model('cache/inswapper_128.onnx')
        self.face_enhancer = gfpgan.GFPGANer(model_path='cache/GFPGANv1.4.pth', upscale=1)
        self.face_analyser = FaceAnalysis(name='buffalo_l')
        self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
    
    def get_faces(self, img_data):
        analysed = self.face_analyser.get(img_data)
        if not analysed:
            print("No faces found")
            return []
        return analysed

    def swap(self, target_image: Path, swap_image: Path) -> Path:
        try:
            target_frame = cv2.imread(str(target_image))
            swap_frame = cv2.imread(str(swap_image))
            target_faces = self.get_faces(target_frame)
            swap_faces = self.get_faces(swap_frame)

            if not target_faces or not swap_faces:
                print("No faces to process.")
                return None

            for i, target_face in enumerate(target_faces):
                try:
                    source_face = swap_faces[i % len(swap_faces)]
                    print("Swapping face")
                    swapped_frame = self.face_swapper.get(target_frame, target_face, source_face, paste_back=True)
                    if swapped_frame is not None:
                        target_frame = swapped_frame
                    else:
                        print("Swapping failed for one face, continuing with others.")
                except Exception as swap_error:
                    print(f"Error during individual face swap: {swap_error}")
                    continue  # Skip to the next face if the current one fails
            
            # save the swapped frame
            cv2.imwrite("swapped.jpg", target_frame)

            return "swapped.jpg"


            print("Enhancing frame")
            enhanced_frame, _, _ = self.face_enhancer.enhance(target_frame, paste_back=True)
            out_path = Path(tempfile.mkdtemp()) / f"{str(int(time.time()))}.jpg"
            cv2.imwrite(str(out_path), enhanced_frame)
            return out_path

        except Exception as e:
            print(f"Error during face swapping: {e}")
            return None


    def predict(
    self,
    prompt: str = Input(description="Prompt to generate images from"),
    image: Path = Input(description="Grayscale input image"),
    gaussian_radius: int = Input(description="Gaussian blur radius", default=7),
) -> str:
        """Run a single prediction on the model"""

        # load image as <class 'PIL.Image.Image'>
        pil_image = Image.open(image)

        # Get poses using the Openpose model
        poses = self.model(pil_image)

        # Generate the initial image based on the prompt and poses
        generator = torch.Generator("cuda").manual_seed(random.randint(0, 1000000000))
        input_prompt = [prompt]
        out = self.pipe(
            input_prompt,
            poses,
            negative_prompt=["naked, boobs, tits, nsfw, monochrome, lowres, bad anatomy, worst quality, low quality"],
            generator=generator,
            num_inference_steps=20,
        )
        out_image = out.images[0]

        # save output image
        out_image = out.images[0]
        out_image.save("output.png")

        # swap faces
        swapped_image = self.swap("output.png", image)

        if swapped_image:
             # convert to base64
            with open(swapped_image, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())

            # return base64 in an array
            return str([encoded_string.decode("utf-8")])
        else:
            with open("output.png", "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())

            return str([encoded_string.decode("utf-8")])
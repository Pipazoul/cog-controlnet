# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image, ImageFilter

import base64
import cv2
import numpy as np
import random
from cog import BasePredictor, Input, Path

MODEL = "lllyasviel/ControlNet"
CONTROLNET = "fusing/stable-diffusion-v1-5-controlnet-openpose"
MODEL_ID = "runwayml/stable-diffusion-v1-5"

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



    def predict(
        self,
        image: Path = Input(description="Grayscale input image")
    ) -> str:
        """Run a single prediction on the model"""

        # load image as <class 'PIL.Image.Image'>
        pil_image = Image.open(image)

        poses = self.model(pil_image)

        #bg black
        BG_COLOR = (0, 0, 0) # black
        MASK_COLOR = (255, 255, 255) # white

        # Create the options that will be used for ImageSegmenter
        base_options = python.BaseOptions(model_asset_path='selfie_multiclass_256x256.tflite')
        options = vision.ImageSegmenterOptions(base_options=base_options,
                                            output_confidence_masks=True)

        # Create the image segmenter
        with vision.ImageSegmenter.create_from_options(options) as segmenter:
            # Read the image to be processed
            #TypeError: a bytes-like object is required, not 'Image'
            image = open(image, 'rb').read()


            # save image base64 to local
            image = np.frombuffer(image, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            image_file_name = 'image.jpg'
            cv2.imwrite(image_file_name, image)

            # Create the MediaPipe image file that will be segmented
            image = mp.Image.create_from_file(image_file_name)

            # Retrieve the masks for the segmented image
            segmentation_result = segmenter.segment(image)
            category_mask = segmentation_result.confidence_masks[3]
            
            #print(category_mask)

            # Generate solid color images for showing the output segmentation mask.
            image_data = image.numpy_view()
            fg_image = np.zeros(image_data.shape, dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros(image_data.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR

            condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
            output_image = np.where(condition, fg_image, bg_image)


            # save output image
            Image.fromarray(output_image).save("mask.png")

        

        generator = torch.Generator("cuda").manual_seed(random.randint(0, 1000000000))
        #generator = [torch.Generator(device="cpu").manual_seed(2)]
        prompt = ["super-hero character, best quality, extremely detailed"]
        out = self.pipe(
            prompt,
            poses,
            negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"],
            generator=generator,
            num_inference_steps=20,
        )

        # save output image
        out_image = out.images[0]
        out_image.save("output.png")

        
        # Load the images
        foreground = Image.open("image.jpg").convert("RGBA")
        background = Image.open("output.png").convert("RGBA")
        mask = Image.open("mask.png").convert("L")  # Convert mask to grayscale

        # Resize foreground to match background image size
        foreground = foreground.resize(background.size)

        # Resize mask to match background image size
        mask = mask.resize(background.size)

        mask = mask.filter(ImageFilter.GaussianBlur(radius=7))


        # Apply the mask to the foreground image
        foreground.putalpha(mask)


        # Composite the images
        result = Image.alpha_composite(background, foreground)

        # Save the result
        result.save("output_with_overlay.png")
        
        # convert to base64
        with open("output_with_overlay.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())

        # return base64 in an array
        return [encoded_string.decode("utf-8")]





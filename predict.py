# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import torch
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from PIL import Image

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
        image: str = Input(description="Grayscale input image")
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        print("input_image", image)
        # Load the image
        image = load_image(image)
        #image = Image.open(image).convert("RGB")

        # resize the image to 512x512
        image = image.resize((512, 512), Image.BICUBIC)
        
        poses = self.model(image)
        

        generator = torch.Generator("cuda").manual_seed(5)
        #generator = [torch.Generator(device="cpu").manual_seed(2)]
        prompt = ["super-hero character, best quality, extremely detailed"]
        out = self.pipe(
            prompt,
            poses,
            negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"],
            generator=generator,
            num_inference_steps=20,
        )
        out.images[0].save("output.png")

        return Path("output.png")

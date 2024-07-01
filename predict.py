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
from PIL import Image, ImageDraw, ImageFilter
import io

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

        # Initialize MediaPipe face detection
        self.face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    
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

    def extract_and_merge_faces(self, input_image: Image, output_image: Image, gaussian_radius: int, face_opacity: float = 0.8) -> Image:
        """
        Extract faces from the input image and merge them with the output image using face landmarks.
        
        Args:
        input_image (PIL.Image): The original input image
        output_image (PIL.Image): The generated output image
        gaussian_radius (int): The radius for Gaussian blur
        face_opacity (float): Opacity of the face overlay (0.0 to 1.0, default is 0.8)

        Returns:
        PIL.Image: The merged image with faces from the input image
        """
        # Create the options for ImageSegmenter
        base_options = python.BaseOptions(model_asset_path='selfie_multiclass_256x256.tflite')
        options = vision.ImageSegmenterOptions(base_options=base_options, output_confidence_masks=True)

        # Create the image segmenter
        with vision.ImageSegmenter.create_from_options(options) as segmenter:
            # Convert PIL images to numpy arrays
            input_np = np.array(input_image)
            output_np = np.array(output_image)

            # Ensure both images have the same size and 3 channels
            target_shape = output_np.shape[:2]
            input_np = cv2.resize(input_np, (target_shape[1], target_shape[0]))
            
            if len(input_np.shape) == 2:
                input_np = cv2.cvtColor(input_np, cv2.COLOR_GRAY2BGR)
            if len(output_np.shape) == 2:
                output_np = cv2.cvtColor(output_np, cv2.COLOR_GRAY2BGR)

            # Create the MediaPipe image that will be segmented
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=input_np)

            # Retrieve the masks for the segmented image
            segmentation_result = segmenter.segment(mp_image)
            category_mask = segmentation_result.confidence_masks[3]  # Assuming index 3 is for face

            # Generate mask
            mask = (category_mask.numpy_view() > 0.2).astype(np.float32)
            mask = cv2.resize(mask, (target_shape[1], target_shape[0]))

            # Apply Gaussian blur to the mask
            mask = cv2.GaussianBlur(mask, (gaussian_radius*2+1, gaussian_radius*2+1), 0)

            # Expand mask to 3 channels
            mask_3channel = np.stack([mask] * 3, axis=-1)

            # Ensure all arrays are float32
            input_np = input_np.astype(np.float32) / 255.0
            output_np = output_np.astype(np.float32) / 255.0
            mask_3channel = mask_3channel.astype(np.float32)

            # Apply face opacity
            mask_3channel *= face_opacity

            # Blend images
            merged = input_np * mask_3channel + output_np * (1 - mask_3channel)

            # Convert back to uint8 and then to PIL Image
            merged_uint8 = (merged * 255).astype(np.uint8)
            merged_pil = Image.fromarray(merged_uint8, 'RGB')

            # Save the merged image
            merged_pil.save("merged.png")

        return merged_pil
    def predict(
    self,
    prompt: str = Input(description="Prompt to generate images from"),
    image: Path = Input(description="Grayscale input image"),
    gaussian_radius: int = Input(description="Gaussian blur radius", default=6),
    face_opacity: float = Input(description="Opacity of the face overlay (0.0 to 1.0)", default=0.7),
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
            negative_prompt=["worst quality, low quality"],
            generator=generator,
            num_inference_steps=20,
        )
        out_image = out.images[0]

        # save output image
        out_image = out.images[0]
        out_image.save("output.png")

        # open the output image
        out_image = Image.open("output.png")


        #merge image
        merged_image = self.extract_and_merge_faces(pil_image, out_image, gaussian_radius, face_opacity)


        # swap faces
        swapped_image = self.swap("merged.png", image)
        #swapped_image = self.swap("output.png", image)

        # correct faces with gfpgan
        swapped_image = self.face_enhancer.enhance("swapped.jpg", paste_back=True)

        # save the final image
        cv2.imwrite("esrghan.png", swapped_image)

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
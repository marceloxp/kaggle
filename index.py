import cv2 as cv
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from diffusers import (AutoPipelineForImage2Image, StableDiffusionControlNetPipeline, ControlNetModel)
import torch
import os

def choose_device(torch_device = None):
    print('...Is CUDA available in your computer?',\
          '\n... Yes!' if torch.cuda.is_available() else "\n... No D': ")
    print('...Is MPS available in your computer?',\
          '\n... Yes' if torch.backends.mps.is_available() else "\n... No D':")

    if torch_device is None:
        if torch.cuda.is_available():
            torch_device = "cuda"
            torch_dtype = torch.float16
        elif torch.backends.mps.is_available() and not torch.cuda.is_available():
            torch_device = "mps"
            torch_dtype = torch.float16
        else:
            torch_device = "cpu"
            torch_dtype = torch.float32

    print("......using ", torch_device)

    return torch_device, torch_dtype

#################
# CONSTANTS
#################

DEFAULT_PROMPT                =  "a beautiful woman, floral bikini"
MODEL                         = "lcm" #"lcm" # or "sdxlturbo"
SDXLTURBO_MODEL_LOCATION      = 'models/sdxl-turbo'
LCM_MODEL_LOCATION            = 'SimianLuo/LCM_Dreamshaper_v7'
CONTROLNET_CANNY_LOCATION     = "lllyasviel/control_v11p_sd15_canny" 
TORCH_DEVICE, TORCH_DTYPE     = choose_device()  
GUIDANCE_SCALE                = 3 # 0 for sdxl turbo (hardcoded already)
INFERENCE_STEPS               = 4 #4 for lcm (high quality) #2 for turbo
DEFAULT_NOISE_STRENGTH        = 0.7 # 0.5 works well too
CONDITIONING_SCALE            = .7 # .5 works well too
GUIDANCE_START                = 0.
GUIDANCE_END                  = 1.
RANDOM_SEED                   = 21
WIDTH                         = 512 #512 #384 #512
HEIGHT                        = 768 #512 #384 #512

def prepare_seed():
    generator = torch.manual_seed(RANDOM_SEED)
    return generator

def convert_numpy_image_to_pil_image(image):
    return Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

def get_result_and_mask(frame, center_x, center_y, width, height):
    "just gets full frame and the mask for cutout"
    
    mask = np.zeros_like(frame)
    mask[center_y:center_y+height, center_x:center_x+width, :] = 255
    cutout = frame[center_y:center_y+height, center_x:center_x+width, :]

    return frame, cutout

def process_lcm(image, lower_threshold = 100, upper_threshold = 100, aperture=3): 
    image = np.array(image)
    image = cv.Canny(image, lower_threshold, upper_threshold,apertureSize=aperture)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    return image

def prepare_lcm_controlnet_pipeline():

    controlnet = ControlNetModel.from_pretrained(CONTROLNET_CANNY_LOCATION, torch_dtype=TORCH_DTYPE, use_safetensors=True)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(LCM_MODEL_LOCATION,\
                                                    controlnet=controlnet, 
                                                    # unet=unet,\
                                                    torch_dtype=TORCH_DTYPE, safety_checker=None).\
                                                to(TORCH_DEVICE)
    return pipeline

def run_lcm(pipeline, ref_image):

    generator = prepare_seed()
    gen_image = pipeline(prompt                        = DEFAULT_PROMPT,
                         num_inference_steps           = INFERENCE_STEPS, 
                         guidance_scale                = GUIDANCE_SCALE,
                         width                         = WIDTH, 
                         height                        = HEIGHT, 
                         generator                     = generator,
                         image                         = ref_image, 
                         controlnet_conditioning_scale = CONDITIONING_SCALE, 
                         control_guidance_start        = GUIDANCE_START, 
                         control_guidance_end          = GUIDANCE_END, 
                        ).images[0]

    return gen_image

def run_lcm_or_sdxl():
    pipeline = prepare_lcm_controlnet_pipeline()
    processor = process_lcm
    run_model = run_lcm

    # Load the image
    image = cv.imread('image.png')

    numpy_image = processor(image)
    pil_image = convert_numpy_image_to_pil_image(numpy_image)
    pil_image = run_model(pipeline, pil_image)

    result_image = cv.cvtColor(np.array(pil_image), cv.COLOR_RGB2BGR)

    # Save the resulting image as output.png
    cv.imwrite("output.png", result_image)

    image = cv.imread('output.png')

    # Converter de BGR para RGB (matplotlib usa RGB)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Exibir a imagem
    plt.imshow(image_rgb)
    plt.axis('off')  # Desativar os eixos
    plt.show()


### RUN SCRIPT
run_lcm_or_sdxl()
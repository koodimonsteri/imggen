from datetime import datetime
import logging
from pathlib import Path

from diffusers import AutoPipelineForText2Image
import torch


logger = logging.getLogger(__name__)


def prompt_to_filename(prompt_str: str):
    cleaned_prompt = prompt_str.rstrip('.').replace('.', ',').lower().replace(' ', '_').replace('"', '')
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'{cleaned_prompt}_{ts_str}.jpg'


def main():

    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")
    n_inference = 4
    prompts = [
        #"Goofy cat wearing a top silk hat.",
        #"Huge cat on top of a mountain.",
        #"Blue colored cat sitting on top of a strawberry.",
        #"Cats sitting around a campfire. One of them is reading a book.",
        #"Happy elephant drinking a glass of water.",
        #"Happy blue elephant drinking a glass of water.",
        #"Cartoony happy blue elephant drinking a glass of water.",
        #"Happy elephant drinking a pint of beer.",
        #"Elephant eating a banana.",
        #"Blue elephant eating a banana.",
        #"Realistic elephant standing on top of a high-rise building.",
        #"Baby elephant swimming in a lake.",
        #"Cat sitting on top of a building and eats a bowl of strawberries",
        #"Baby elephant in a bowl.",
        #"Elephant sitting in a cardboard box.",
        #"Baby elephant sint next to a graffiti that says 'RONSU'.",
        #"Blue baby elephant in a cardboard box.",
        #"Blue baby elephant sideways next to a graffiti text 'RONSU'.",
        #"Blue baby elephant sideways next to a graffiti 'RONSU'.",
        #"Blue baby elephant sideways next to a graffiti with text 'RONSU'.",
        #"Cat sitting on top of a blue colored elephant.",# Background is large mountain.",
        #"Majestic cat sitting on top of a blue colored elephant.",# Background is large mountain.",
        #"Cat sitting on top of a blue colored elephant.",# Background is large mountain.",
        "Cat sitting on top of realistic blue colored elephant. Background is graffiti that reads 'RONSU'",
        "Cat sitting on top of shiny blue colored elephant. Background is graffiti that reads 'RONSU'",
        "Cat sitting on top of artistic blue colored elephant. Background is graffiti that reads 'RONSU'",
        "Cat sitting on top of majestic blue colored elephant. Background is graffiti that reads 'RONSU'",
        "Cat sitting on top of cartoony blue colored elephant. Background is graffiti that reads 'RONSU'",


        #"Blue baby elephant sideways next to a graffiti that reads 'RONSU'.",        #######
        #"Blue elephant sideways next to a graffiti that reads 'RONSU'.",             #######
        #"Blue elephant next to a graffiti that reads 'RONSU'.",                      #######
        "Cartoony blue elephant sideways next to a graffiti that reads 'RONSU'.",    #######
        "Majestic blue elephant sideways next to a graffiti that reads 'RONSU'.",    #######
        "Blue elephant sideways next to a graffiti that reads 'RONSU'.",    #######

        #"Blue baby elephant sideways next to a graffiti that says 'RONSU'.",
        #"Blue baby elephant sideways with graffiti text 'RONSU'.",
        #"Blue baby elephant next to a graffiti with text 'RONSU'.",
        #"Blue baby elephant next to a graffiti that reads 'RONSU'.",
        #"Blue colored cartoon elephant. Background is graffiti with text \"RONSU\".",
        #"Finlayson elephant walking on a grass.",
        #"Happy blue elephant next to a graffiti with text \"RONSU\".",
        #"Cartoony blue elephant sitting in a bowl.",
        #"Realistic blue elephant sitting in a bowl.",
        #"Blue elephant sitting in a bowl.",
    ]
    inference_folder = f'inf{n_inference}'
    out_path = Path(r'./images') / inference_folder
    logger.info('Running with %s inference steps', n_inference)
    for _ in range(0, 10):
        for prompt_msg in prompts:
            logger.info('Generating image with prompt: %s', prompt_msg)
            image = pipe(prompt=prompt_msg, num_inference_steps=n_inference, guidance_scale=0.0).images[0]
            fname = prompt_to_filename(prompt_msg)
            logger.info('Saving image to: %s', out_path / fname)
            image.save(out_path / fname)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()

import io
import logging

import asyncio
from concurrent.futures import ThreadPoolExecutor
from diffusers import AutoPipelineForText2Image
import discord
import torch


logger = logging.getLogger(__name__)

DISCORD_TOKEN = ''
SERVER_NAME = 'BotTesting'
MSG_PREFIX = 'catgen:'


intents = discord.Intents(messages=True, guilds=True, members=True, message_content=True)


def prompt_to_filename(prompt_str: str):
    return prompt_str.rstrip('.').replace('.', ',').lower().replace(' ', '_') + '.jpg'


def valid_generate_prompt(prompt):
    return any([x in prompt.lower() for x in ['cat', 'elephant', 'ronsu']])


class CatGenClient(discord.Client):
    """
    Works fine when generating one image at a time.
    Image generation sometimes fails if piping multiple prompts too quickly.
    Should probably decouple image generation from discord bot.
    """
    def __init__(self):
        self.catgenerator = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
        self.catgenerator.to("cuda")
        self.executor = ThreadPoolExecutor(max_workers=5)
        super().__init__(intents=intents)

    async def on_ready(self):
        guild = discord.utils.get(self.guilds, name=SERVER_NAME)
        logger.info("%s is connected to the following guild:", self.user)
        logger.info("%s(id: %s)", guild.name, guild.id)

    async def on_message(self, message):
        if message.author == self.user:
            return

        msg = message.content
        if not msg.startswith(MSG_PREFIX):
            logger.info('Not bot command')
            return

        parsed = msg.split(':')
        if len(parsed) != 2:
            await message.channel.send("Invalid command: prompt can't have ':' characters.")
            logger.info('Incorrect command')

        prompt = parsed[1]
        logger.info('Received prompt: %s', prompt)

        if prompt == 'help':
            await message.channel.send('Generate images of cats :)')

        elif valid_generate_prompt(prompt):
            logger.info('Generate and upload image')
            await message.channel.send(f'Generating image with prompt: {prompt}')
            await self.generate_image(message, prompt)

        else:
            logger.info('Prompt must include cat or elephant.')
            await message.channel.send("Prompt must include 'cat' or 'elephant'")

    async def generate_image(self, message, prompt):
        new_prompt = prompt.replace('ronsu', 'blue colored elephant').replace('Ronsu', 'Blue colored elephant')

        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(self.executor, lambda: generate_image_blocking(new_prompt, self.catgenerator))

        with io.BytesIO() as image_binary:
            image.save(image_binary, 'PNG')
            image_binary.seek(0)
            filename = prompt_to_filename(prompt)
            logger.info('Upload image to discord.')
            await message.channel.send(f"{prompt}", file=discord.File(fp=image_binary, filename=filename))


def generate_image_blocking(prompt, generator):
    return generator(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]


def main():
    catgen = CatGenClient()
    catgen.run(DISCORD_TOKEN)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    main()
    
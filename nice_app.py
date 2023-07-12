#!/usr/bin/env python3
import asyncio
import functools
import io
from typing import Callable
import requests

# import replicate  # very nice API to run AI models; see https://replicate.com/

from nicegui import ui
from nicegui.events import UploadEventArguments




async def io_bound(callback: Callable, *args: any, **kwargs: any):
    '''Makes a blocking function awaitable; pass function as first parameter and its arguments as the rest'''
    return await asyncio.get_event_loop().run_in_executor(None, functools.partial(callback, *args, **kwargs))


async def generate_audio():
    # transcription.text = 'Composing...'
    uri = 'http://localhost:8080/v2/models/music_model/infer'
    prediction = await io_bound(requests.post, url=uri, prompts=prompts.value, seconds=seconds.value)
    music = prediction.json()['outputs'][0]['data']


async def generate_image():
    painting.text = 'Composing...'
    uri = 'http://localhost:8080/v2/models/image_model/infer'
    prediction = await io_bound(requests.post, url=uri, prompt=prompt.value)
    shape = prediction.json()['outputs'][0]['shape']
    data  = prediction.json()['outputs'][0]['data']
    image = prediction.array(data, dtype=np.uint8).reshape(shape)
    image.source = prediction[0]

async def generate_text():
    image.source = 'https://dummyimage.com/600x400/ccc/000000.png&text=building+image...'
    model = replicate.models.get('stability-ai/stable-diffusion')
    version = model.versions.get('db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf')
    prediction = await io_bound(version.predict, prompt=prompt.value)
    image.source = prediction[0]


# User Interface
with ui.row().style('gap:10em'):
    with ui.column():
        ui.label('OpenAI Whisper (voice transcription)').classes('text-2xl')
        ui.upload(on_upload=transcribe).style('width: 20em')
        transcription = ui.label().classes('text-xl')
    with ui.column():
        ui.label('Stable Diffusion (image generator)').classes('text-2xl')
        prompt = ui.input('prompt').style('width: 20em')
        ui.button('Generate', on_click=generate_image).style('width: 15em')
        image = ui.image().style('width: 60em')

ui.run()
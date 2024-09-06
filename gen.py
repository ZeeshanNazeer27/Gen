import torch
import streamlit as st
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image

@st.cache_resource
def load():
    model_id = "stabilityai/stable-diffusion-2"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32)
    pipe = pipe.to("cpu")  
    return pipe

st.title("Astronaut Riding a Horse on Mars - Image Generator")
prompt = st.text_input("Enter your prompt for the image:", "a photo of an astronaut riding a horse on mars")

if st.button("Generate Image"):
    pipe = load()
    
    with st.spinner('Generating image...'):
        image = pipe(prompt).images[0]
    st.image(image, caption="Generated Image", use_column_width=True)

import gradio as gr
from aura_sr import AuraSR
from PIL import Image
import os

# Initialize AuraSR model
aura_sr = AuraSR.from_pretrained()

def upscale_image(input_image):
    # Ensure input_image is a PIL Image
    if not isinstance(input_image, Image.Image):
        input_image = Image.fromarray(input_image)
    
    # Print input size for debugging
    print(f"Input size: {input_image.size}")
    
    # Upscale using AuraSR (4x upscaling, no forced resize)
    upscaled_image = aura_sr.upscale_4x(input_image)
    
    # Print output size for debugging
    print(f"Output size: {upscaled_image.size}")
    
    # Save upscaled image for download
    output_path = "upscaled_output.png"
    upscaled_image.save(output_path)
    
    return upscaled_image, output_path

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AuraSR Image Upscaler")
    gr.Markdown("Drop an image, click Upscale, and download the 4x upscaled result.")
    
    input_image = gr.Image(type="pil", label="Input Image")
    upscale_button = gr.Button("Upscale")
    output_image = gr.Image(label="Upscaled Image")
    download_button = gr.File(label="Download Upscaled Image")
    
    upscale_button.click(
        fn=upscale_image,
        inputs=input_image,
        outputs=[output_image, download_button]
    )

# Launch the interface
demo.launch()
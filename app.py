import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import gradio as gr
from transformers import AutoModelForImageSegmentation
from cv_pipeline import GrabCutPipeline

def model_pipeline(birefnet: torch.nn.Module,
                   image: Image) -> Image:
    image_size = (512, 512)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_images = transform_image(image).unsqueeze(0)
    if torch.cuda.is_available():
        input_images = input_images.to('cuda')

    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    return image

def process_image(method: str,
                  img: Image) -> Image:
    image = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    if method == "GrabCut":
        grabcut_processor = GrabCutPipeline(image)
        grabcut_processor.process_image()
        output_image = grabcut_processor.output
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

        return Image.fromarray(output_image)

    elif method == "BiRefNet":
        birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
        if torch.cuda.is_available():
            birefnet.to('cuda')

        return model_pipeline(birefnet, img)

interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Radio(["GrabCut", "BiRefNet"], label="Background Removal Method"),
        gr.Image(type="pil", label="Upload Image")
    ],
    outputs=gr.Image(type="pil", label="Output Image"),
    title="Background Removal App",
    description="Choose between GrabCut and BiRefNet for background removal and upload your image."
)

if __name__ == "__main__":
    interface.launch()

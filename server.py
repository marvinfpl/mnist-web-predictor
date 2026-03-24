from PIL import Image, ImageOps
from torchvision import transforms
import torch
import io, base64
import time
from model import MnistClassifier
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np

model = MnistClassifier()
model.load_state_dict(torch.load("./model/mnist_classifier.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/mnist", response_class=HTMLResponse)
async def render_mnist():
    with open("mnist.html") as f:
        return f.read()

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()

    if 'image' not in data or "," not in data["image"]:
        return {"error": "Invalid image data"}

    image_b64 = data["image"].split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGBA")

    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    bg.paste(image, mask=image.split()[3])
    bg = bg.convert("L")

    bg = ImageOps.invert(bg)

    bbox = bg.getbbox()
    if bbox:
        bg = bg.crop(bbox)

    bg.thumbnail((140, 140), Image.LANCZOS)
    bg.thumbnail((56, 56), Image.LANCZOS)
    bg.thumbnail((28, 28), Image.LANCZOS)
    
    # binarization

    arr = np.array(bg, dtype=np.float32)
    arr = (arr > 50).astype(np.float32) * 255
    bg = Image.fromarray(arr.astype(np.uint8))

    # center the digit

    bg.thumbnail((20, 20), Image.LANCZOS)
    new_image = Image.new("L", (28,28), 0)
    x_offset = (28 - bg.width)//2
    y_offset = (28 - bg.height)//2
    new_image.paste(bg, (x_offset, y_offset))

    img = new_image

    filename = f"./images/resized_{time.time() * 1000}.png"
    img.save(filename, format="PNG")

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()

    response = {"prediction": pred, "confidence_score": round(confidence * 100, 1)}
    return response
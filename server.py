from PIL import Image, ImageChops
from torchvision import transforms
import torch
import io, base64
from model import MnistClassifier
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

template = Jinja2Templates(directory=".")

@app.get("/mnist", response_class=HTMLResponse)
async def render_mnist():
    with open("mnist.html") as f:
        return f.read()

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    image_b64 = data["image"].split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("L").resize((28, 28))
    image = ImageChops.invert(image)
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    response = {"prediction": pred}
    return response
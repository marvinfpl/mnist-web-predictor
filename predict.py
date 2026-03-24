import sys
from PIL import Image, ImageChops
from torchvision import transforms
import torch
import io, base64
from model import MnistClassifier
from flask import Flask, request, jsonify

app = Flask(__name__)

model = MnistClassifier()
model.load_state_dict(torch.load("./model/mnist_classifier.pth", map_location="cpu"))
model.eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

@app.route("/predict", method=["POST"])
def predict():
    data = request.get_json()
    image_b64 = data["image"].split(",")[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("L").resize((28, 28))
    image = ImageChops.invert(image)
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    return jsonify({"prediction": pred})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import MnistClassifier

epochs = 15
batch_size = 128
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5,)),
])
    
train = datasets.MNIST(root=".", train=True, transform=transform, download=True)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
test = datasets.MNIST(root=".", train=False, transform=transform)
test_loader = DataLoader(test, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MnistClassifier()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.train()

for epoch in tqdm(range(epochs), desc=False):
    total_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch+1}, Total loss: {(total_loss/len(train_loader)):.4f} ")


model.eval()

total = 0
correct = 0
loss = 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        total += labels.size(0)
        outputs = model(inputs)

        loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

print(f" --- Total loss: {loss:.4f}, Accuracy: {(correct / total):.4f} --- ")

torch.save(model.state_dict(), "./model/mnist_classifier.pth")
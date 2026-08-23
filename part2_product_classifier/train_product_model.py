# train_product_model.py
# Train a simple ResNet-18 transfer-learning model on Fashion-MNIST.

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Fashion-MNIST downloads automatically the first time.
train_data = datasets.FashionMNIST(
    root="data", train=True, download=True, transform=transform
)

test_data = datasets.FashionMNIST(
    root="data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

print("Training images:", len(train_data))
print("Test images:", len(test_data))

# Load pretrained ResNet-18.
try:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
except:
    model = models.resnet18(pretrained=True)

# Freeze the pretrained backbone.
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace ImageNet's 1000-class head with a 10-class Fashion-MNIST head.
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# Train only the new classifier head.
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        "Epoch", epoch + 1,
        "Loss:", round(total_loss / len(train_loader), 4)
    )

# Evaluate on the test set.
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

accuracy = correct / total
print("Test accuracy:", round(accuracy, 4))

# Save the trained model.
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/product_classifier.pt")

print("Model saved: models/product_classifier.pt")

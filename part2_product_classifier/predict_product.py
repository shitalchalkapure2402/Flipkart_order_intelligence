# Load the saved model and classify one sample image.

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the same ResNet-18 structure used during training.
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)

# Load the trained model.
model.load_state_dict(
    torch.load(
        "models/product_classifier.pt",
        map_location=device
    )
)

model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Change this filename if you want to test another sample.
image_path = "data/sample_images/07_sneaker.png"

image = Image.open(image_path).convert("L")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    probabilities = torch.softmax(output, dim=1)
    confidence, prediction = probabilities.max(1)

print("Image:", image_path)
print("Predicted category:", class_names[prediction.item()])
print("Confidence:", round(confidence.item(), 4))

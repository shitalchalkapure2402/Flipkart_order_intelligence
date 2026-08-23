
# Evaluate the saved Fashion-MNIST model.
# This code creates:
# 1. Confusion matrix
# 2. Per-class precision
# 3. Per-class recall
# 4. Top 2 actual confusion pairs

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# 1. Class names
# -------------------------------------------------

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# 2. Device
# -------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# 
# 3. Same preprocessing used during training
# -------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# 4. Load Fashion-MNIST test data
# -------------------------------------------------

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print("Test images:", len(test_data))

# 
# 5. Create the same ResNet-18 model
# -------------------------------------------------

model = models.resnet18(weights=None)

# The final layer has 10 classes.
model.fc = nn.Linear(    model.fc.in_features, 10)

# 6. Load your already-trained model
# -------------------------------------------------

model.load_state_dict(
    torch.load("models/product_classifier.pt", map_location=device)
        )

model = model.to(device)
model.eval()

print("Saved model loaded successfully.")

# 7. Get predictions
# -------------------------------------------------

actual_labels = []
predicted_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        actual_labels.extend(
            labels.numpy()
        )

        predicted_labels.extend(
            predictions.cpu().numpy()
        )

# 8. Confusion Matrix
# -------------------------------------------------

cm = confusion_matrix(
    actual_labels,
    predicted_labels
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("Rows = Actual class")
print("Columns = Predicted class\n")

print(cm)

# 9. Per-class Precision
# -------------------------------------------------

precision = precision_score(
    actual_labels,
    predicted_labels,
    average=None,
    labels=list(range(10)),
    zero_division=0
)

print("\n" + "=" * 60)
print("PER-CLASS PRECISION")
print("=" * 60)

for i in range(10):

    print(
        class_names[i],
        ":",
        round(precision[i], 4)
    )

# 10. Per-class Recall
# -------------------------------------------------

recall = recall_score(
    actual_labels,
    predicted_labels,
    average=None,
    labels=list(range(10)),
    zero_division=0
)

print("\n" + "=" * 60)
print("PER-CLASS RECALL")
print("=" * 60)

for i in range(10):

    print(
        class_names[i],
        ":",
        round(recall[i], 4)
    )

# 11. Find top 2 actual confusion pairs
# -------------------------------------------------

# Make a copy so we can remove the diagonal.
# Diagonal values are correct predictions.
confusion_pairs = cm.copy()

for i in range(10):
    confusion_pairs[i, i] = 0

# Find the largest confusion values.
pairs = []

for actual_class in range(10):

    for predicted_class in range(10):

        count = confusion_pairs[
            actual_class,
            predicted_class
        ]

        if count > 0:

            pairs.append(
                (
                    count,
                    actual_class,
                    predicted_class
                )
            )

# Sort from largest confusion to smallest.
pairs.sort(reverse=True)

print("\n" + "=" * 60)
print("TOP 2 ACTUAL CONFUSION PAIRS")
print("=" * 60)

for count, actual_class, predicted_class in pairs[:2]:

    print(
        f"Actual: {class_names[actual_class]} "
        f"--> Predicted: {class_names[predicted_class]} "
        f"| Count: {count}"
    )
# export_sample_images.py
# Download Fashion-MNIST and save 5 real test images as PNG files.

from pathlib import Path
from torchvision import datasets

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True
)

sample_folder = Path("data/sample_images")
sample_folder.mkdir(parents=True, exist_ok=True)

class_names = [
    "tshirt_top", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle_boot"
]

# Save one real image from five different classes.
used_classes = set()

for index in range(len(test_data)):
    image, label = test_data[index]

    if label not in used_classes:
        filename = f"{label:02d}_{class_names[label]}.png"
        image.save(sample_folder / filename)
        print("Saved:", sample_folder / filename)

        used_classes.add(label)

    if len(used_classes) == 5:
        break

print("Done. Check: data/sample_images/")

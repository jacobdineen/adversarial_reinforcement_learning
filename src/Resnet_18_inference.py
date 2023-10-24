# -*- coding: utf-8 -*-
# Import Libraries
import matplotlib.pyplot as plt
import torch

# from src.classifiers import ResidualBlock, ResNet
from src.utils import load_model, preprocess_image

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


# Inference
def score(model, input_image_tensor, classes):
    input_image_tensor = input_image_tensor.to(DEVICE)
    with torch.no_grad():
        output = model(input_image_tensor)
        _, predicted = torch.max(output.data, 1)
    return classes[predicted[0]]


# Main Function
def main():
    model = load_model()
    image_path = "/home/src/images/resnet_cat.jpg"
    input_image_tensor, img = preprocess_image(image_path)
    prediction = score(model, input_image_tensor, CLASSES)
    plt.imshow(img)
    plt.title(f"Predicted class: {prediction}")
    plt.show()
    print(f"Predicted class: {prediction}")


if __name__ == "__main__":
    main()

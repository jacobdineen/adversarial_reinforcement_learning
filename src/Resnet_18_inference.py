# -*- coding: utf-8 -*-
# Import Libraries
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

# from src.classifiers import ResidualBlock, ResNet
from src.utils import load_model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Configuration Setup
def config_setup():
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    return DEVICE, classes


# Image Preprocessing
def preprocess_image(image_path):
    img = Image.open(image_path)
    transform_img = transforms.Compose(
        [transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    img.show()
    input_image_tensor = transform_img(img).unsqueeze(0)
    return input_image_tensor, img


# Inference
def score(model, input_image_tensor, classes):
    input_image_tensor = input_image_tensor.to(DEVICE)
    with torch.no_grad():
        output = model(input_image_tensor)
        _, predicted = torch.max(output.data, 1)
    return classes[predicted[0]]


# Main Function
def main():
    _, classes = config_setup()
    model = load_model()
    image_path = "/home/src/images/resnet_cat.jpg"
    input_image_tensor, img = preprocess_image(image_path)
    prediction = score(model, input_image_tensor, classes)
    plt.imshow(img)
    plt.title(f"Predicted class: {prediction}")
    plt.show()
    print(f"Predicted class: {prediction}")


if __name__ == "__main__":
    main()

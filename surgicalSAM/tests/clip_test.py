from transformers import CLIPTokenizer, CLIPProcessor, CLIPModel
import torch


# print all available models
print(
    "Available models:",
    CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
)

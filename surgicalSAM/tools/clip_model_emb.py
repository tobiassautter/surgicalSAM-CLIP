from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionLayer(nn.Module):
    def __init__(self, input_dim=512, output_dim=256):
        super(ProjectionLayer, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.normalize(self.projection(x), p=2, dim=-1)


def get_emb(output_dim=256):
    # Load the CLIP model and processor
    model_name = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_name)
    clip_processor = CLIPProcessor.from_pretrained(model_name)

    # Define the surgical instruments
    instruments = [
        "Bipolar Forceps",
        "Prograsp Forceps",
        "Large Needle Driver",
        "Monopolar Curved Scissors",
        "Ultrasound Probe",
        "Suction Instrument",
        "Clip Applier",
    ]

    # Tokenize and encode the instrument descriptions
    inputs = clip_processor(text=instruments, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)

    # Normalize the embeddings
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

    # Project CLIP embeddings to the desired dimension
    projection_layer = ProjectionLayer(
        input_dim=text_features.size(1), output_dim=output_dim
    )
    projected_features = projection_layer(text_features)

    return projected_features

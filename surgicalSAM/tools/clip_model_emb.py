from os import environ

# Disable parallelism in tokenizers to avoid fork issues
environ["TOKENIZERS_PARALLELISM"] = "false"

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

class CLIPEmbeddings:
    def __init__(self, model_name="openai/clip-vit-base-patch32", output_dim=256):
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)
        
        self.projection_layer = ProjectionLayer(input_dim=512, output_dim=output_dim)

        self.output_dim = output_dim

        self.instrument_details = [
            "Bipolar Forceps",
            "Prograsp Forceps",
            "Large Needle Driver",
            "Monopolar Curved Scissors",
            "Ultrasound Probe",
            "Suction Instrument",
            "Clip Applier",
        ]

    def get_embeddings(self):
        # Tokenize and encode the detailed descriptions
        inputs = self.clip_processor(
            text=self.instrument_details, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)

        # Normalize the embeddings
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
        # Project the embeddings to the desired dimension
        text_features = self.projection_layer(text_features)

        return text_features

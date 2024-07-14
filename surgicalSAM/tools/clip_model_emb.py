# from os import environ

# # Disable parallelism in tokenizers to avoid fork issues
# environ["TOKENIZERS_PARALLELISM"] = "false"

# from transformers import CLIPProcessor, CLIPModel
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class ProjectionLayer(nn.Module):
#     def __init__(self, input_dim=512, output_dim=256):
#         super(ProjectionLayer, self).__init__()
#         self.projection = nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         return F.normalize(self.projection(x), p=2, dim=-1)


# class CLIPEmbeddings:
#     """
#     Class to get embeddings from the CLIP model

#     Args:
#     - model_name: The name of the CLIP model to use
#     - output_dim: The dimension of the output embeddings
#     - instrument_details: The detailed descriptions of the instruments to get embeddings for

#     Returns:
#     - text_features: The embeddings for the detailed descriptions of the instruments
#     """

#     def __init__(
#         self, model_name="openai/clip-vit-base-patch32", output_dim=256, pI=[]
#     ):
#         self.clip_model = CLIPModel.from_pretrained(model_name)
#         self.clip_processor = CLIPProcessor.from_pretrained(model_name)

#         self.projection_layer = ProjectionLayer(input_dim=512, output_dim=output_dim)

#         self.output_dim = output_dim

#         self.instrument_details = [
#             "Bipolar Forceps",
#             "Prograsp Forceps",
#             "Large Needle Driver",
#             "Monopolar Curved Scissors",
#             "Ultrasound Probe",
#             "Suction Instrument",
#             "Clip Applier",
#         ]

#         if len(pI) > 0:
#             self.instrument_details = pI

#     def get_embeddings(self):
#         # Tokenize and encode the detailed descriptions
#         inputs = self.clip_processor(
#             text=self.instrument_details, return_tensors="pt", padding=True
#         )
#         # with torch.no_grad():
#         text_features = self.clip_model.get_text_features(**inputs)

#         # Normalize the embeddings
#         text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
#         # Project the embeddings to the desired dimension
#         text_features = self.projection_layer(text_features)

#         return text_features

#     # get embedding 256 dim


from os import environ
import torch
from transformers import CLIPProcessor, CLIPModel, CLIPConfig

environ["TOKENIZERS_PARALLELISM"] = "false"


class CLIPEmbeddings:
    """
    Class to get embeddings from the CLIP model

    Args:
    - model_name: The name of the CLIP model to use
    - output_dim: The dimension of the output embeddings
    - instrument details: The detailed descriptions of the instruments to get embeddings for

    Returns:
    - text_features: The embeddings for the detailed descriptions of the instruments
    """

    def __init__(
        self, model_name="openai/clip-vit-base-patch32", output_dim=256, pI=[]
    ):
        # Setting the projection dimension in the CLIP configuration
        self.clip_config = CLIPConfig(projection_dim=output_dim)
        self.clip_model = CLIPModel(self.clip_config)
        self.clip_processor = CLIPProcessor.from_pretrained(model_name)

        # Set the model to evaluation mode to disable training specific layers like dropout
        self.clip_model.eval()

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

        if len(pI) > 0:
            print("Using custom instrument details")
            self.instrument_details = pI

    def get_embeddings(self):
        # Tokenize and encode the detailed descriptions
        inputs = self.clip_processor(
            text=self.instrument_details, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            # Directly retrieve projected text features
            text_features = self.clip_model.get_text_features(**inputs)

        # The embeddings are automatically normalized within the model
        return text_features


# Example use of the CLIPEmbeddings class
# clip_embeddings = CLIPEmbeddings()
# text_embeddings = clip_embeddings.get_embeddings()
# print(
#     text_embeddings.shape
# )  # Should output something like [7, 256] based on the number of instruments

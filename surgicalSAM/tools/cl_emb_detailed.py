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
        self.projection_layer = ProjectionLayer(
            input_dim=512, output_dim=output_dim
        )  # Adjust input_dim if necessary
        self.output_dim = output_dim

        # Define the surgical instruments
        self.instrument_details = {
            # Fill this dictionary as previously detailed
            # "background": [
            #     "background",
            #     "the area represented by the background tissues.",
            #     "background tissues are the tissues that surround the surgical site.",
            # ],
            # "instrument": [
            #     "instrument",
            #     "the area represented by the instrument.",
            #     "instruments in endoscopic surgery typically exhibit elongated designs, specialized tips or jaws for specific functions, ergonomic handles for precise control, and insulated shafts to minimize energy transmission or tissue damage.",
            # ],
            "bipolar_forceps": [
                "bipolar forceps",
                "the area represented by the bipolar forceps.",
                "bipolar forceps have a slim, elongated tweezer-like design with opposing tips, are silver-colored, made from high-quality metal, and feature an insulated shaft for controlled energy application.",
            ],
            "prograsp_forceps": [
                "prograsp forceps",
                "the area represented by the prograsp forceps.",
                "prograsp forceps possess curved scissor-like handles, specialized grasping tips with interlocking jaws, a ratcheting mechanism, and color-coded markings for easy identification during surgery.",
            ],
            "large_needle_driver": [
                "large needle driver",
                "the area represented by the large needle driver.",
                "large needle drivers feature elongated handles, sturdy gripping surfaces, a curved or straight jaw tip for securely holding needles, and a locking mechanism to ensure precision and control.",
            ],
            "monopolar_curved_scissors": [
                "monopolar curved scissors",
                "the area represented by the monopolar curved scissors.",
                "monopolar curved scissors showcase elongated handles, curved cutting edges for precise dissection, and an insulated shaft, allowing controlled application of electrical energy for cutting and coagulation.",
            ],
            "ultrasound_probe": [
                "ultrasound probe",
                "the area represented by the ultrasound probe.",
                "ultrasound probes feature a long, slender handle, a small transducer head for producing ultrasound waves, and a flexible cable connecting the probe to the ultrasound machine for real-time imaging guidance.",
            ],
            "suction_instrument": [
                "suction instrument",
                "the area represented by the suction instrument.",
                "suction instruments appear as elongated tubes with a narrow, hollow tip for fluid and debris removal, connected to a handle and tubing system for vacuum generation and precise control during the procedure.",
            ],
            "clip_applier": [
                "clip applier",
                "the area represented by the clip applier.",
                "clip appliers feature elongated handles, a shaft with a specialized tip for holding and releasing clips, and a mechanism to advance and deploy the clips precisely for secure tissue or vessel closure.",
            ],
        }

    def get_embeddings(self):
        # Prepare detailed descriptions for embedding generation
        detailed_descriptions = [
            desc[-1] for desc in self.instrument_details.values()
        ]  # Assuming the last description is the most detailed

        # Tokenize and encode the detailed descriptions
        inputs = self.clip_processor(
            text=detailed_descriptions, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)

        # Normalize the embeddings
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

        # Project CLIP embeddings to the desired dimension
        projected_features = self.projection_layer(text_features)

        return projected_features

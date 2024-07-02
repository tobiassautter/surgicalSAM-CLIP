from transformers import CLIPProcessor, CLIPModel
import torch


def get_emb():
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

    return text_features

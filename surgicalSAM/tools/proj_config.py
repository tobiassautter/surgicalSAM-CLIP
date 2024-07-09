from transformers import CLIPModel, CLIPConfig, CLIPProcessor
import torch

# Define the custom configuration
clip_config = CLIPConfig(projection_dim=256)
clip_model = CLIPModel(clip_config)

# Assuming you have CLIPProcessor for handling inputs
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Example usage: Get embeddings at 256 dimensions directly
def get_clip_embeddings(texts):
    inputs = clip_processor(text=texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = clip_model.get_text_features(**inputs)
    # Outputs now are already at 256 dimensions
    return outputs


# Example text inputs
texts = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver"]
embeddings = get_clip_embeddings(texts)
print(embeddings.shape)  # Should show (number of texts, 256)

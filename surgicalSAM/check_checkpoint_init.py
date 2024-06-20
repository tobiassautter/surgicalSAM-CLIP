import torch


def load_and_inspect_checkpoint(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Print out all parameter names and their dimensions
    print("Listing all parameters and their dimensions in the checkpoint:")
    for name, param in checkpoint.items():
        print(f"{name}: {param.size()}")


# Relative path from your script's location to the checkpoint
checkpoint_path = "../ckp/medSAM/medsam_vit_b.pth"

# Call the function
load_and_inspect_checkpoint(checkpoint_path)

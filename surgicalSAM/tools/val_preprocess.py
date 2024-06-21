import sys

sys.path.append("../..")
import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
import torch
from torch.nn import functional as F
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide


def set_torch_image(transformed_mask):
    input_mask = preprocess(transformed_mask)  # pad to 1024
    return input_mask


def set_mask(mask):
    """Transform the mask to the form expected by SAM, the transformed mask will be used to generate class embeddings
    Adapated from set_image in the official code of SAM https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py
    """
    input_mask = ResizeLongestSide(1024).apply_image(mask)
    input_mask_torch = torch.as_tensor(input_mask)
    input_mask_torch = input_mask_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    input_mask = set_torch_image(input_mask_torch)

    return input_mask


def preprocess(x):
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)

    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


# Initialize SAM model
vit_mode = "b"
sam_checkpoint = "../../ckp/medSAM/medsam_vit_b.pth"
sam = sam_model_registry[f"vit_{vit_mode}"](checkpoint=sam_checkpoint)
print("SAM model loaded: ", sam_checkpoint)
sam.cuda()
predictor = SamPredictor(sam)

# Define data locations
dataset_name = "endovis_2018"
data_root_dir = f"../../data/{dataset_name}"
mask_dir = osp.join(data_root_dir, "val", "binary_annotations")
frame_dir = osp.join(data_root_dir, "val", "images")

frame_list = [
    os.path.join(os.path.basename(subdir), filename)
    for subdir, _, files in os.walk(frame_dir)
    for filename in files
    if files
]
mask_list = [
    os.path.join(os.path.basename(subdir), filename)
    for subdir, _, files in os.walk(mask_dir)
    for filename in files
    if files
]
# print files in frame and mask directories
[print(f) for f in frame_list[:]]
[print(f) for f in mask_list[:]]
# print total number of frames and masks
print("Total frames: ", len(frame_list))
print("Total masks: ", len(mask_list))
# print path to both
print("Frame path: ", frame_dir)
print("Mask path: ", mask_dir)

# Processing frames and masks
for n, frame_name in enumerate(frame_list):

    frame_path = osp.join(frame_dir, frame_name)
    print(f"Processing frame {n+1}/{len(frame_list)}: {frame_name}")

    # Find matching masks
    masks_name = [
        mask for mask in mask_list if mask.split("_")[0] == frame_name.split(".")[0]
    ]
    masks_name = sorted(masks_name)

    # Load and process masks
    original_masks = []
    for mask_name in masks_name:
        mask_path = osp.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.uint8(mask == 255)
        mask = Image.fromarray(mask)
        original_masks.append(mask)

    masks = [np.asarray(mask) * 255 for mask in original_masks]

    # set frame vars
    original_frame = cv2.imread(frame_path)
    original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    original_frame = Image.fromarray(original_frame)

    # obtain SAM feature of the augmented frame
    original_frame = np.asarray(original_frame)

    predictor.set_image(original_frame)
    feat = predictor.features.squeeze().permute(1, 2, 0)
    feat = feat.cpu().numpy()

    # Save the frame SAM feature
    feat_save_dir = osp.join(
        data_root_dir, "val/sam_features_b", frame_name.split(".")[0] + ".npy"
    )

    print(f"Saving frame feature to {feat_save_dir}")

    os.makedirs(osp.dirname(feat_save_dir), exist_ok=True)
    np.save(feat_save_dir, feat)

    # Process and save masks and embeddings
    for mask, mask_name in zip(masks, masks_name):

        # process augmented_masks to the same shape and format as the image
        zeros = np.zeros_like(mask)
        mask_processed = np.stack((mask, zeros, zeros), axis=-1)
        mask_processed = set_mask(mask_processed)
        mask_processed = F.interpolate(
            mask_processed, size=torch.Size([64, 64]), mode="bilinear"
        )
        mask_processed = mask_processed.squeeze()[0]
        # print(mask_processed.shape) and path to mask
        print("mask_processed shape: ", mask_processed.shape)
        print("mask_path: ", mask_path)

        # if the augmented mask after processing does not have any foreground objects, then skip this mask
        if (True in (mask_processed > 0)) == False:
            continue

        # compute the class embedding using frame SAM feature and processed mask
        class_embedding = feat[mask_processed > 0]
        class_embedding = class_embedding.mean(0).squeeze()

        # save the augmented mask and the computed class embedding
        class_embedding_save_dir = osp.join(
            data_root_dir,
            f"val/class_embeddings_{vit_mode}",
            mask_name.replace("png", "npy"),
        )
        os.makedirs(osp.dirname(class_embedding_save_dir), exist_ok=True)

        np.save(class_embedding_save_dir, class_embedding)

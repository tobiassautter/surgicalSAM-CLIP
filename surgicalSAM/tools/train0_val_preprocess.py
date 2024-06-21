import sys

sys.path.append("../..")
import os
import os.path as osp
import cv2
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image
import random
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.nn import functional as F
import argparse


def set_mask(mask):
    """Transform the mask to the form expected by SAM, the transformed mask will be used to generate class embeddings
    Adapated from set_image in the official code of SAM https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py
    """
    input_mask = ResizeLongestSide(1024).apply_image(mask)
    input_mask_torch = torch.as_tensor(input_mask)
    input_mask_torch = input_mask_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    input_mask = set_torch_image(input_mask_torch)

    return input_mask


def set_torch_image(transformed_mask):
    input_mask = preprocess(transformed_mask)  # pad to 1024
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


# define the SAM model
vit_mode = "b"
if vit_mode == "b":
    sam_checkpoint = "../../ckp/medSAM/medsam_vit_b.pth"
sam = sam_model_registry[f"vit_{vit_mode}"](checkpoint=sam_checkpoint)
sam.cuda()
predictor = SamPredictor(sam)

# define data
dataset_name = "endovis_2018"
data_root_dir = f"../../data/{dataset_name}"
folder = ["train", "val"]


mask_dir = osp.join(data_root_dir, "train", "0", "binary_annotations")
frame_dir = osp.join(data_root_dir, "train", "0", "images")

frame_list = [
    os.path.join(os.path.basename(subdir), file)
    for subdir, _, files in os.walk(frame_dir)
    for file in files
    if files
]
mask_list = [
    os.path.join(os.path.basename(subdir), file)
    for subdir, _, files in os.walk(mask_dir)
    for file in files
    if files
]


# go though each frame one by one
for n, frame_name in enumerate(frame_list):

    frame_path = osp.join(frame_dir, frame_name)
    original_frame = cv2.imread(frame_path)
    original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    original_frame = Image.fromarray(original_frame)

    # read all the original masks (without any augmentation) of the current frame and organise them into a list
    masks_name = [
        mask for mask in mask_list if mask.split("_")[0] == frame_name.split(".")[0]
    ]
    masks_name = sorted(masks_name)

    original_masks = []
    for mask_name in masks_name:
        mask_path = osp.join(mask_dir, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.uint8(mask == 255)
        mask = Image.fromarray(mask)
        original_masks.append(mask)

    # set seed for reproducibility
    random.seed(version)
    torch.manual_seed(version)
    torch.cuda.manual_seed(version)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(version)

    frame = np.asarray(frame)
    masks = [np.asarray(mask) * 255 for mask in masks]

    # obtain SAM feature of the augmented frame
    predictor.set_image(frame)
    feat = predictor.features.squeeze().permute(1, 2, 0)
    feat = feat.cpu().numpy()

    # save the augmented frame and its SAM feature
    if dataset_name == "endovis_2018":
        save_dir = osp.join(data_root_dir, "train", str(version))

    frame_save_dir = osp.join(save_dir, "images", frame_name)
    feat_save_dir = osp.join(
        save_dir, f"sam_features_{vit_mode}", frame_name.split(".")[0] + "npy"
    )
    os.makedirs(osp.dirname(frame_save_dir), exist_ok=True)
    os.makedirs(osp.dirname(feat_save_dir), exist_ok=True)

    frame = Image.fromarray(frame)
    frame.save(frame_save_dir)
    np.save(feat_save_dir, feat)

    # go through each augmented mask
    for mask, mask_name in zip(masks, masks_name):

        # process augmented_masks to the same shape and format as the image
        zeros = np.zeros_like(mask)
        mask_processed = np.stack((mask, zeros, zeros), axis=-1)
        mask_processed = set_mask(mask_processed)
        mask_processed = F.interpolate(
            mask_processed, size=torch.Size([64, 64]), mode="bilinear"
        )
        mask_processed = mask_processed.squeeze()[0]

        # if the augmented mask after processing does not have any foreground objects, then skip this mask
        if (True in (mask_processed > 0)) == False:
            continue

        # compute the class embedding using frame SAM feature and processed mask
        class_embedding = feat[mask_processed > 0]
        class_embedding = class_embedding.mean(0).squeeze()

        # save the augmented mask and the computed class embedding
        mask_save_dir = osp.join(save_dir, "binary_annotations", mask_name)
        class_embedding_save_dir = osp.join(
            save_dir,
            f"class_embeddings_{vit_mode}",
            mask_name.replace("png", "npy"),
        )
        os.makedirs(osp.dirname(mask_save_dir), exist_ok=True)
        os.makedirs(osp.dirname(class_embedding_save_dir), exist_ok=True)

        mask = Image.fromarray(mask)
        mask.save(mask_save_dir)
        np.save(class_embedding_save_dir, class_embedding)

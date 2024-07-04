import sys

sys.path.append("..")
import os
import os.path as osp
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import Endovis18Dataset, Endovis17Dataset
from segment_anything import sam_model_registry
from model_clip import Learnable_Prototypes, Prototype_Prompt_Encoder
from utils import (
    print_log,
    create_binary_masks,
    create_endovis_masks,
    eval_endovis,
    read_gt_endovis_masks,
)
from model_forward import model_forward_function
from loss import DiceLoss
from pytorch_metric_learning import losses
from datetime import datetime

# Exponantial LR
# from torch.optim.lr_scheduler import ExponentialLR
# from torch.optim.lr_scheduler import LinearLR

## logger
import wandb_logger

## import clip_model_emb.py
import tools.cl_emb_detailed as cl_em_dt


print("======> Set Parameters for Training")
dataset_name = "endovis_2018"
fold = 0
thr = 0
# project def
seed = 123  # 666
data_root_dir = f"../../SurgicalSAM/data/{dataset_name}"
batch_size = 16  # 32  # 32
vit_mode = "h"  # "h"
num_epochs = 100  # 500
lr = 0.002  # 0.001
num_workers = 2  # 4

# for logger
w_project_name = "surgicalSAM - Endovis 2018 - SSAM-clip-full"
c_loss_temp = 0.07

# set seed for reproducibility -----------------------------------------------------
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

# load dataset-specific parameters ------------------------------------------------
print("======> Load Dataset-Specific Parameters")

num_tokens = 2
val_dataset = Endovis18Dataset(
    data_root_dir=data_root_dir, mode="val", vit_mode=vit_mode
)

gt_endovis_masks = read_gt_endovis_masks(data_root_dir=data_root_dir, mode="val")
save_dir = "./work_dirs/endovis_2018/"


val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
# load sam model -------------------------------------------------------------------
print("======> Load SAM")
sam_checkpoint = "../ckp/sam/sam_vit_h_4b8939.pth"

print("Checkpoint: ", sam_checkpoint)
model_type = "vit_h_no_image_encoder"

sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](
    checkpoint=sam_checkpoint
)

sam_prompt_encoder.cuda()
sam_decoder.cuda()

# load clip embeddings -------------------------------------------------------------
print("======> Load CLIP Embeddings")
clip_embeddings_handler = cl_em_dt.CLIPEmbeddings()
fixed_prototypes = clip_embeddings_handler.get_embeddings()

# Setup the learnable prototypes and encoder with fixed prototypes ----------------
print("======> Load Prototypes and Prototype-based Prompt Encoder")
learnable_prototypes_model = Learnable_Prototypes(
    num_classes=7, feat_dim=256, clip_embeddings=fixed_prototypes
).cuda()

protoype_prompt_encoder = Prototype_Prompt_Encoder(
    feat_dim=256,
    hidden_dim_dense=128,
    hidden_dim_sparse=128,
    size=64,
    num_tokens=num_tokens,
).cuda()

# Ensure all parameters except the decoder are not trainable -----------------------
for param in learnable_prototypes_model.parameters():
    param.requires_grad = False
for param in protoype_prompt_encoder.parameters():
    param.requires_grad = False
for param in sam_decoder.parameters():
    param.requires_grad = True


#
# with open(sam_checkpoint, "rb") as f:
#     state_dict = torch.load(f)
#     sam_pn_embeddings_weight = {
#         k.split("prompt_encoder.point_embeddings.")[-1]: v
#         for k, v in state_dict.items()
#         if k.startswith("prompt_encoder.point_embeddings") and ("0" in k or "1" in k)
#     }
#     sam_pn_embeddings_weight_ckp = {
#         "0.weight": torch.concat(
#             [sam_pn_embeddings_weight["0.weight"] for _ in range(num_tokens)], dim=0
#         ),
#         "1.weight": torch.concat(
#             [sam_pn_embeddings_weight["1.weight"] for _ in range(num_tokens)], dim=0
#         ),
#     }

#     protoype_prompt_encoder.pn_cls_embeddings.load_state_dict(
#         sam_pn_embeddings_weight_ckp
#     )

# Load the state dictionary only if needed
with open(sam_checkpoint, "rb") as f:
    state_dict = torch.load(f)
    # Filter only needed parts if necessary, assuming you need to load something specific
    prototype_related_state_dict = {
        k: v for k, v in state_dict.items() if "prototype" in k
    }
    protoype_prompt_encoder.load_state_dict(prototype_related_state_dict, strict=False)


# optimiser and loss --------------------------------------------------------------
print("======> Define Optmiser and Loss")
# Adjusting the optimizer to only update the decoder parameters
optimizer = torch.optim.Adam(
    sam_decoder.parameters(),
    lr=lr,
    weight_decay=0.0001,
)

# Continue using the same loss functions as before
seg_loss_model = DiceLoss().cuda()
contrastive_loss_model = losses.NTXentLoss(temperature=c_loss_temp).cuda()


# make dirs and logs -------------------------------------------------------------

print("======> Set Saving Directories and Logs")
os.makedirs(save_dir, exist_ok=True)
log_file = osp.join(save_dir, "log_clip_SS_full.txt")
print_log(dataset_name, log_file)

print("======> Start Training and Validation")
best_challenge_iou_val = -100.0

# for logging
print("======> Initialize wandb")
wandb_logger.init(
    project=w_project_name,
    config={
        "learning_rate": lr,
        "architecture": "SSAM - Clip Full",
        "dataset": dataset_name,
        "epochs": num_epochs,
        "temperature": c_loss_temp,
        "batch_size": batch_size,
        "fold": fold,
        "learning_rate": lr,
        "seed": seed,
        "vit_mode": vit_mode,
    },
)


# training and validation loop ----------------------------------------------------
for epoch in range(num_epochs):
    if epoch % 2 == 0:
        version = 0
    else:
        version = int((epoch % 80 + 1) / 2)

    train_dataset = Endovis18Dataset(
        data_root_dir=data_root_dir,
        mode="train",
        vit_mode=vit_mode,
        version=version,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    sam_decoder.train()
    for sam_feats, _, cls_ids, masks, class_embeddings in train_dataloader:
        sam_feats, cls_ids, masks, class_embeddings = (
            sam_feats.cuda(),
            cls_ids.cuda(),
            masks.cuda(),
            class_embeddings.cuda(),
        )
        # Print cls_ids to verify correct batch data
        print(f"Current batch cls_ids: {cls_ids}")

        # Ensuring the use of correct prototypes for each class ID
        prototypes = fixed_prototypes[cls_ids]
        print(f"Prototypes selected for current batch: {prototypes.shape}")

        preds, _ = model_forward_function(
            protoype_prompt_encoder,
            sam_prompt_encoder,
            sam_decoder,
            sam_feats,
            prototypes,
            cls_ids,
        )

        # Output dimensions of predictions for sanity check
        print(f"Predictions shape: {preds.shape}")

        seg_loss = seg_loss_model(preds, masks / 255)
        contrastive_loss = contrastive_loss_model(
            prototypes,
            torch.arange(prototypes.size(0)).cuda(),
            ref_emb=class_embeddings,
            ref_labels=cls_ids,
        )
        # Total loss is the sum of segmentation and contrastive loss ----------------------------
        loss = seg_loss + contrastive_loss
        print(
            f"Segmentation Loss: {seg_loss.item()}, Contrastive Loss: {contrastive_loss.item()}"
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # validation
    binary_masks = dict()
    protoype_prompt_encoder.eval()
    sam_decoder.eval()
    # learnable_prototypes_model.eval()

    with torch.no_grad():
        # During validation (if using the same logic as training)
        for sam_feats, mask_names, cls_ids, _, _ in val_dataloader:
            sam_feats, cls_ids = sam_feats.cuda(), cls_ids.cuda()
            # Using fixed prototypes based on class IDs
            prototypes = fixed_prototypes[cls_ids]

            preds, preds_quality = model_forward_function(
                protoype_prompt_encoder,
                sam_prompt_encoder,
                sam_decoder,
                sam_feats,
                prototypes,
                cls_ids,
            )

            binary_masks = create_binary_masks(
                binary_masks, preds, preds_quality, mask_names, thr
            )

    endovis_masks = create_endovis_masks(binary_masks, 1024, 1280)
    endovis_results = eval_endovis(endovis_masks, gt_endovis_masks)

    # print validation results in log
    print_log(
        f"Validation - Epoch: {epoch}/{num_epochs-1}; IoU_Results: {endovis_results} ",
        log_file,
    )
    # print timestamp in log
    print_log(
        f"Timestamp: {datetime.now()} -----------------------------------------------",
        log_file,
    )
    # log results to wandb
    wandb_logger.log_results(endovis_results)

    # save the model with the best challenge IoU
    if endovis_results["challengIoU"] > best_challenge_iou_val:
        best_challenge_iou_val = endovis_results["challengIoU"]

        torch.save(
            {
                "prototype_prompt_encoder_state_dict": protoype_prompt_encoder.state_dict(),
                "sam_decoder_state_dict": sam_decoder.state_dict(),
                "prototypes_state_dict": learnable_prototypes_model.state_dict(),
            },
            osp.join(save_dir, "model_ckp_SSAM_clip.pth"),
        )

        print_log(
            f"Best Challenge IoU: {best_challenge_iou_val:.4f} at Epoch {epoch}",
            log_file,
        )

# close wandb
wandb_logger.close()

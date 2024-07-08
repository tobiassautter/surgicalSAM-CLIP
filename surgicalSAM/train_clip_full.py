import sys
import os
import os.path as osp

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import Endovis18Dataset, Endovis17Dataset
from segment_anything import sam_model_registry
from model import Learnable_Prototypes, Prototype_Prompt_Encoder
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
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LinearLR

## logger
import wandb_logger

## import clip_model_emb.py
import tools.clip_model_emb as cl_em_dt

print("======> Process Arguments")
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="endovis_2018",
    choices=["endovis_2018", "endovis_2017"],
    help="specify dataset",
)
parser.add_argument(
    "--fold",
    type=int,
    default=0,
    choices=[0, 1, 2, 3],
    help="specify fold number for endovis_2017 dataset",
)
args = parser.parse_args()

print("======> Set Parameters for Training")
dataset_name = args.dataset
fold = args.fold
thr = 0
seed = 123  # 666
data_root_dir = f"../../SurgicalSAM/data/{dataset_name}"
# data_root_dir = osp.join("..", "data", dataset_name)
print("Data Root Dir: ", data_root_dir)
batch_size = 24  # 32  # 32
vit_mode = "h"  # "h"
use_agumentation = True
# for logger
w_project_name = "surgicalSAM - Endovis 2018 - SSAM-clip-full"
c_loss_temp = 0.07
log_data = True
n_w = 8


# set seed for reproducibility
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)


print("======> Load Dataset-Specific Parameters")
if "18" in dataset_name:
    num_tokens = 2
    val_dataset = Endovis18Dataset(
        data_root_dir=data_root_dir, mode="val", vit_mode="h"
    )

    gt_endovis_masks = read_gt_endovis_masks(data_root_dir=data_root_dir, mode="val")
    num_epochs = 500  # 500
    lr = 0.0008  # 0.001
    save_dir = osp.join("work_dirs", "endovis_2018")
    # "./work_dirs/endovis_2018/"


val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=n_w
)


print("======> Load SAM")
if vit_mode == "h":
    # sam_checkpoint = "../ckp/sam/sam_vit_h_4b8939.pth"
    sam_checkpoint = osp.join("..", "ckp", "sam", "sam_vit_h_4b8939.pth")
print("Checkpoint: ", sam_checkpoint)
model_type = "vit_h_no_image_encoder"

sam_prompt_encoder, sam_decoder = sam_model_registry[model_type](
    checkpoint=sam_checkpoint
)


sam_prompt_encoder.cuda()
sam_decoder.cuda()

for name, param in sam_prompt_encoder.named_parameters():
    param.requires_grad = False

# only train decoder
for name, param in sam_decoder.named_parameters():
    param.requires_grad = True

# load clip embeddings
print("======> Load CLIP Embeddings")
# clip_emb = clip_model_emb.get_emb(output_dim=256)
feat_dim = 256
clip_embeddings_handler = cl_em_dt.CLIPEmbeddings(output_dim=feat_dim)
clip_emb = clip_embeddings_handler.get_embeddings().cuda()

print("======> Load Prototypes and Prototype-based Prompt Encoder")
learnable_prototypes_model = Learnable_Prototypes(
    num_classes=7, feat_dim=feat_dim, clip_embeddings=clip_emb
).cuda()

protoype_prompt_encoder = Prototype_Prompt_Encoder(
    feat_dim=feat_dim,
    hidden_dim_dense=128,
    hidden_dim_sparse=128,
    size=64,
    num_tokens=num_tokens,
).cuda()

# freeze the prototype encoder
for name, param in learnable_prototypes_model.named_parameters():
    param.requires_grad = False
# freeze the prompt encoder
for name, param in protoype_prompt_encoder.named_parameters():
    param.requires_grad = False


with open(sam_checkpoint, "rb") as f:
    state_dict = torch.load(f)
    sam_pn_embeddings_weight = {
        k.split("prompt_encoder.point_embeddings.")[-1]: v
        for k, v in state_dict.items()
        if k.startswith("prompt_encoder.point_embeddings") and ("0" in k or "1" in k)
    }
    sam_pn_embeddings_weight_ckp = {
        "0.weight": torch.concat(
            [sam_pn_embeddings_weight["0.weight"] for _ in range(num_tokens)], dim=0
        ),
        "1.weight": torch.concat(
            [sam_pn_embeddings_weight["1.weight"] for _ in range(num_tokens)], dim=0
        ),
    }

    protoype_prompt_encoder.pn_cls_embeddings.load_state_dict(
        sam_pn_embeddings_weight_ckp
    )

print("======> Define Optmiser and Loss")
seg_loss_model = DiceLoss().cuda()
# contrastive_loss_model = losses.NTXentLoss(temperature=c_loss_temp).cuda()  # 0.07

optimiser = torch.optim.Adam(
    [
        # {"params": learnable_prototypes_model.parameters()},
        # {"params": protoype_prompt_encoder.parameters()},
        {"params": sam_decoder.parameters()},
    ],
    lr=lr,
    weight_decay=0.0001,  # 0.0001,
)


print("======> Set Saving Directories and Logs")
os.makedirs(save_dir, exist_ok=True)
log_file = osp.join(save_dir, "log_clip_SS.txt")
print_log(str(args), log_file)

print("======> Start Training and Validation")
best_challenge_iou_val = -100.0

# for logging
if log_data:
    print("======> Initialize wandb")
    wandb_logger.init(
        project=w_project_name,
        config={
            "learning_rate": lr,
            "architecture": "SSAM - orig",
            "dataset": dataset_name,
            "epochs": num_epochs,
            "temperature": c_loss_temp,
            "batch_size": batch_size,
            "num workers": n_w,
        },
    )

for epoch in range(num_epochs):

    # choose the augmentation version to use for the current epoch
    if use_agumentation:
        if epoch % 2 == 0:
            version = 0
        else:
            version = int((epoch % 80 + 1) / 2)
    else:
        version = 0

    if "18" in dataset_name:
        train_dataset = Endovis18Dataset(
            data_root_dir=data_root_dir,
            mode="train",
            vit_mode=vit_mode,
            version=version,
        )

    # elif "17" in dataset_name:
    #     train_dataset = Endovis17Dataset(
    #         data_root_dir=data_root_dir,
    #         mode="train",
    #         fold=fold,
    #         vit_mode=vit_mode,
    #         version=version,
    #     )
    # print(train_dataset.__len__())

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_w
    )

    # training
    protoype_prompt_encoder.eval()  # .train()
    sam_decoder.train()
    learnable_prototypes_model.eval()  # .train()

    for sam_feats, _, cls_ids, masks, class_embeddings in train_dataloader:

        sam_feats = sam_feats.cuda()
        cls_ids = cls_ids.cuda()
        masks = masks.cuda()
        class_embeddings = class_embeddings.cuda()

        prototypes = learnable_prototypes_model()

        preds, _ = model_forward_function(
            protoype_prompt_encoder,
            sam_prompt_encoder,
            sam_decoder,
            sam_feats,
            prototypes,
            cls_ids,
        )

        # compute loss
        # contrastive_loss = contrastive_loss_model(
        #     prototypes,
        #     torch.tensor([i for i in range(1, prototypes.size()[0] + 1)]).cuda(),
        #     ref_emb=class_embeddings,
        #     ref_labels=cls_ids,
        # )
        seg_loss = seg_loss_model(preds, masks / 255)

        loss = seg_loss  # + contrastive_loss

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    # validation
    binary_masks = dict()
    protoype_prompt_encoder.eval()
    sam_decoder.eval()
    learnable_prototypes_model.eval()

    with torch.no_grad():
        prototypes = learnable_prototypes_model()

        for sam_feats, mask_names, cls_ids, _, _ in val_dataloader:

            sam_feats = sam_feats.cuda()
            cls_ids = cls_ids.cuda()

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
    if log_data:
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
            osp.join(save_dir, "model_ckp_SSAM.pth"),
        )

        print_log(
            f"Best Challenge IoU: {best_challenge_iou_val:.4f} at Epoch {epoch}",
            log_file,
        )

# close wandb
if log_data:
    wandb_logger.close()

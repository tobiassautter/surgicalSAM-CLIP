import numpy as np 
import cv2 
import torch 
import os 
import os.path as osp 
import re
# find system path characters
import logging

sep = str(os.sep)

def create_binary_masks(binary_masks, preds, preds_quality, mask_names, thr, dataset_name="endovis_2018"):
    """Gather the predicted binary masks of different frames and classes into a dictionary, mask quality is also recorded

    Returns:
        dict: a dictionary containing all predicted binary masks organized based on sequence, frame, and mask name
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = preds.to(device)
    preds_quality = preds_quality.to(device)
    
    pred_masks = (preds > thr).int()
    sep = osp.sep

    for pred_mask, mask_name, pred_quality in zip(pred_masks, mask_names, preds_quality):        
        if dataset_name == "endovis_2017":
            seq_name = osp.dirname(mask_name)
            frame_name = osp.basename(mask_name)
        else:  # Default to endovis_2018 or other datasets with similar naming conventions
            seq_name = mask_name.split(sep)[0]
            seq_name = seq_name.replace("/", sep)
            frame_name = osp.basename(mask_name).split("_")[0]

        if seq_name not in binary_masks.keys():
            binary_masks[seq_name] = dict()
        
        if frame_name not in binary_masks[seq_name].keys():
            binary_masks[seq_name][frame_name] = list()
            
        binary_masks[seq_name][frame_name].append({
            "mask_name": mask_name,
            "mask": pred_mask,
            "mask_quality": pred_quality.item()
        })
        
    return binary_masks

def create_endovis_masks(binary_masks, H, W, dataset_name="endovis_2018"):
    """Given the dictionary containing all predicted binary masks, compute final prediction of each frame and organize the prediction masks into a dictionary.
       H - height of image 
       W - width of image
    
    Returns: a dictionary containing one prediction mask for each frame with the frame name as key and its predicted mask as value; 
             For each frame, the binary masks of different classes are combined into a single prediction mask;
             The prediction mask for each frame is a map with each value representing the class id for the pixel;
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    endovis_masks = dict()
    
    for seq in binary_masks.keys():
        for frame in binary_masks[seq].keys():
            endovis_mask = torch.zeros((H, W), dtype=torch.int32, device=device)
    
            binary_masks_list = binary_masks[seq][frame]
            binary_masks_list = sorted(binary_masks_list, key=lambda x: x["mask_quality"])

            for binary_mask in binary_masks_list:
                mask_name = binary_mask["mask_name"]
                predicted_label = int(re.search(r"class(\d+)", mask_name).group(1))
                mask = binary_mask["mask"].to(device)  # Ensure mask is on the correct device
                endovis_mask[mask == 1] = predicted_label

            endovis_mask = endovis_mask.cpu().numpy().astype(int)  # Move to CPU and convert to numpy array

            if dataset_name == "endovis_2017":
                frame = frame.split("_")[0]
                seq_path = osp.join(seq, f"{frame}.png")
            elif dataset_name == "endovis_2018":  # Default to endovis_2018 or other datasets with similar naming conventions
                seq_path = osp.join(seq, "{}.png".format(frame))
            else:
                assert False, "Dataset name not recognized"

            endovis_masks[seq_path] = endovis_mask
    
    return endovis_masks


# Older versions without cuda acceler   ation
# def eval_endovis(endovis_masks, gt_endovis_masks):
#     """Given the predicted masks and groundtruth annotations, predict the challenge IoU, IoU, mean class IoU, and the IoU for each class
        
#       ** The evaluation code is taken from the official evaluation code of paper: ISINet: An Instance-Based Approach for Surgical Instrument Segmentation
#       ** at https://github.com/BCV-Uniandes/ISINet
      
#     Args:
#         endovis_masks (dict): the dictionary containing the predicted mask for each frame 
#         gt_endovis_masks (dict): the dictionary containing the groundtruth mask for each frame 

#     Returns:
#         dict: a dictionary containing the evaluation results for different metrics 
#     """

#     endovis_results = dict()
#     num_classes = 7
    
#     all_im_iou_acc = []
#     all_im_iou_acc_challenge = []
#     cum_I, cum_U = 0, 0
#     class_ious = {c: [] for c in range(1, num_classes+1)}
    
#     for file_name, prediction in endovis_masks.items():
       
#         full_mask = gt_endovis_masks[file_name]
        
#         im_iou = []
#         im_iou_challenge = []
#         target = full_mask.numpy()
#         gt_classes = np.unique(target)
#         gt_classes.sort()
#         gt_classes = gt_classes[gt_classes > 0] 
#         if np.sum(prediction) == 0:
#             if target.sum() > 0: 
#                 all_im_iou_acc.append(0)
#                 all_im_iou_acc_challenge.append(0)
#                 for class_id in gt_classes:
#                     class_ious[class_id].append(0)
#             continue

#         gt_classes = torch.unique(full_mask)
#         # loop through all classes from 1 to num_classes 
#         for class_id in range(1, num_classes + 1): 

#             current_pred = (prediction == class_id).astype(np.float64)
#             current_target = (full_mask.numpy() == class_id).astype(np.float64)

#             if current_pred.astype(np.float64).sum() != 0 or current_target.astype(np.float64).sum() != 0:
#                 i, u = compute_mask_IU_endovis(current_pred, current_target)     
#                 im_iou.append(i/u)
#                 cum_I += i
#                 cum_U += u
#                 class_ious[class_id].append(i/u)
#                 if class_id in gt_classes:
#                     im_iou_challenge.append(i/u)
        
#         if len(im_iou) > 0:
#             all_im_iou_acc.append(np.mean(im_iou))
#         if len(im_iou_challenge) > 0:
#             all_im_iou_acc_challenge.append(np.mean(im_iou_challenge))

#     # calculate final metrics
#     final_im_iou = cum_I / (cum_U + 1e-15)
#     mean_im_iou = np.mean(all_im_iou_acc)
#     mean_im_iou_challenge = np.mean(all_im_iou_acc_challenge)

#     final_class_im_iou = torch.zeros(9)
#     cIoU_per_class = []
#     for c in range(1, num_classes + 1):
#         final_class_im_iou[c-1] = torch.tensor(class_ious[c]).float().mean()
#         cIoU_per_class.append(round((final_class_im_iou[c-1]*100).item(), 3))
        
#     mean_class_iou = torch.tensor([torch.tensor(values).float().mean() for c, values in class_ious.items() if len(values) > 0]).mean().item()
    
#     endovis_results["challengIoU"] = round(mean_im_iou_challenge*100,3)
#     endovis_results["IoU"] = round(mean_im_iou*100,3)
#     endovis_results["mcIoU"] = round(mean_class_iou*100,3)
#     endovis_results["mIoU"] = round(final_im_iou*100,3)
    
#     endovis_results["cIoU_per_class"] = cIoU_per_class
    
#     return endovis_results

# def compute_mask_IU_endovis(masks, target):
#     """compute iou used for evaluation
#     """
#     assert target.shape[-2:] == masks.shape[-2:]
#     temp = masks * target
#     intersection = temp.sum()
#     union = ((masks + target) - temp).sum()
#     return intersection, union


def compute_mask_IU_endovis(pred_mask, target_mask):
    """Compute Intersection and Union (IoU) for the given masks."""
    pred_mask = pred_mask.bool()
    target_mask = target_mask.bool()
    intersection = (pred_mask & target_mask).sum().float()
    union = (pred_mask | target_mask).sum().float()
    return intersection.item(), union.item()

def eval_endovis(endovis_masks, gt_endovis_masks):
    """Given the predicted masks and groundtruth annotations, predict the challenge IoU, IoU, mean class IoU, and the IoU for each class.
    
    Args:
        endovis_masks (dict): the dictionary containing the predicted mask for each frame 
        gt_endovis_masks (dict): the dictionary containing the groundtruth mask for each frame 

    Returns:
        dict: a dictionary containing the evaluation results for different metrics 
    """

    endovis_results = dict()
    num_classes = 7
    
    all_im_iou_acc = []
    all_im_iou_acc_challenge = []
    cum_I, cum_U = 0, 0
    class_ious = {c: [] for c in range(1, num_classes+1)}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for file_name, prediction in endovis_masks.items():
        prediction = torch.tensor(prediction, device=device)
        #full_mask = torch.tensor(gt_endovis_masks[file_name]).clone().detach().to(device)
        full_mask = torch.as_tensor(gt_endovis_masks[file_name], device=device).clone().detach().requires_grad_(False)
        
        im_iou = []
        im_iou_challenge = []
        target = full_mask.clone()
        gt_classes = torch.unique(target)
        gt_classes = gt_classes[gt_classes > 0]
        if prediction.sum().item() == 0:
            if target.sum().item() > 0: 
                all_im_iou_acc.append(0)
                all_im_iou_acc_challenge.append(0)
                for class_id in gt_classes:
                    class_ious[class_id.item()].append(0)
            continue

        # loop through all classes from 1 to num_classes 
        for class_id in range(1, num_classes + 1): 
            current_pred = (prediction == class_id).float()
            current_target = (full_mask == class_id).float()

            if current_pred.sum().item() != 0 or current_target.sum().item() != 0:
                i, u = compute_mask_IU_endovis(current_pred, current_target)     
                im_iou.append(i/u)
                cum_I += i
                cum_U += u
                class_ious[class_id].append(i/u)
                if class_id in gt_classes:
                    im_iou_challenge.append(i/u)
        
        if len(im_iou) > 0:
            all_im_iou_acc.append(np.mean(im_iou))
        if len(im_iou_challenge) > 0:
            all_im_iou_acc_challenge.append(np.mean(im_iou_challenge))

    # calculate final metrics
    final_im_iou = cum_I / (cum_U + 1e-15)
    mean_im_iou = np.mean(all_im_iou_acc)
    mean_im_iou_challenge = np.mean(all_im_iou_acc_challenge)

    final_class_im_iou = torch.zeros(num_classes + 2, device=device)
    cIoU_per_class = []
    for c in range(1, num_classes + 1):
        if len(class_ious[c]) > 0:
            final_class_im_iou[c-1] = torch.tensor(class_ious[c], device=device).mean()
            cIoU_per_class.append(round((final_class_im_iou[c-1].item() * 100), 3))
        else:
            cIoU_per_class.append(0)
        
    mean_class_iou = torch.tensor([torch.tensor(values, device=device).mean() for c, values in class_ious.items() if len(values) > 0]).mean().item()
    
    endovis_results["challengIoU"] = round(mean_im_iou_challenge*100, 3)
    endovis_results["IoU"] = round(mean_im_iou*100, 3)
    endovis_results["mcIoU"] = round(mean_class_iou*100, 3)
    endovis_results["mIoU"] = round(final_im_iou*100, 3)
    
    endovis_results["cIoU_per_class"] = cIoU_per_class
    
    return endovis_results



def read_gt_endovis_masks(data_root_dir = "../data/endovis_2018",
                          mode = "val", 
                          fold = None):
    
    """Read the annotation masks into a dictionary to be used as ground truth in evaluation.

    Returns:
        dict: mask names as key and annotation masks as value 
    """
    gt_endovis_masks = dict()
    
    if "2018" in data_root_dir:
        gt_endovis_masks_path = osp.join(data_root_dir, mode, "annotations")
        for seq in os.listdir(gt_endovis_masks_path):
            for mask_name in os.listdir(osp.join(gt_endovis_masks_path, seq)):
                full_mask_name = osp.join(seq, mask_name)
                mask = torch.from_numpy(cv2.imread(osp.join(gt_endovis_masks_path, full_mask_name),cv2.IMREAD_GRAYSCALE))
                gt_endovis_masks[full_mask_name] = mask
                
    elif "2017" in data_root_dir:
        if fold == "all":
            seqs = [1,2,3,4,5,6,7,8]
            
        elif fold in [0,1,2,3]:
            fold_seq = {0: [1, 3],
                        1: [2, 5],
                        2: [4, 8],
                        3: [6, 7]}
            
            seqs = fold_seq[fold]
        
        gt_endovis_masks_path = osp.join(data_root_dir, "0", "annotations")
        print(gt_endovis_masks_path)
        
        for seq in seqs:
            for mask_name in os.listdir(osp.join(gt_endovis_masks_path, f"seq{seq}")):
                
                full_mask_name = osp.join("seq{}".format(seq), mask_name) #f"{seq}/{mask_name}"
                
                mask = torch.from_numpy(cv2.imread(osp.join(gt_endovis_masks_path, full_mask_name),cv2.IMREAD_GRAYSCALE))
                gt_endovis_masks[full_mask_name] = mask
            
            
    return gt_endovis_masks


def print_log(str_to_print, log_file):
    """Print a string and meanwhile write it to a log file
    """
    print(str_to_print)
    with open(log_file, "a") as file:
        file.write(str_to_print+"\n")
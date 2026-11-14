#!/usr/bin/env python3
"""
Validation script for SAM3 LoRA model
Loads saved weights and runs validation with detailed debugging
"""

import os
import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image as PILImage
import contextlib

# SAM3 Imports
from sam3.model_builder import build_sam3_image_model
from sam3.model.model_misc import SAM3Output
from sam3.train.loss.loss_fns import IABCEMdetr, Boxes, Masks, CORE_LOSS_KEY
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
from sam3.model.box_ops import box_xywh_to_xyxy
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights, count_parameters

from torchvision.transforms import v2

# Import evaluation modules
from sam3.eval.cgf1_eval import CGF1Evaluator, COCOCustom
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils
from sam3.train.masks_ops import rle_encode

# Import SAM3's NMS
from sam3.perflib.nms import nms_masks

class COCOSegmentDataset(Dataset):
    """Dataset class for COCO format segmentation data"""
    def __init__(self, data_dir, split="train", all_class_queries=False):
        """
        Args:
            data_dir: Root directory containing train/valid/test folders
            split: One of 'train', 'valid', 'test'. None means data_dir IS the
                   split directory (no extra nesting).
            all_class_queries: Emit a query for EVERY class of the split on
                every image, with an empty target list for absent classes.
                Gives the presence head negative prompts, which is what makes
                per-class (classification-aware) scoring possible.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir if split is None else self.data_dir / split
        self.all_class_queries = bool(all_class_queries)

        # Load COCO annotations
        ann_file = self.split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)

        # Build index: image_id -> image info
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = sorted(list(self.images.keys()))

        # Build index: image_id -> list of annotations
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Load categories (keep the REAL ids so predictions/GT can be scored
        # per class instead of collapsing everything into a single category).
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

        used_cat_ids = {ann.get('category_id') for ann in self.coco_data['annotations']}
        self.eval_prompt_categories = [
            (self.categories[cid].lower(), int(cid))
            for cid in sorted(c for c in used_cat_ids if c in self.categories)
        ]

        self.object_categories_by_index = [
            {i: int(a.get('category_id', -1))
             for i, a in enumerate(self.img_to_anns.get(img_id, []))
             if a.get('bbox') is not None}
            for img_id in self.image_ids
        ]

        print(f"Loaded COCO dataset: {split} split")
        print(f"  Images: {len(self.image_ids)}")
        print(f"  Annotations: {len(self.coco_data['annotations'])}")
        print(f"  Categories: {self.categories}")
        print(f"  Eval prompts: {[p for p, _ in self.eval_prompt_categories]}")
        print(f"  All-class queries: {self.all_class_queries}")

        self.resolution = 1008
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        # Load image
        img_path = self.split_dir / img_info['file_name']
        pil_image = PILImage.open(img_path).convert("RGB")
        orig_w, orig_h = pil_image.size

        # Resize image
        pil_image = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)

        # Transform to tensor
        image_tensor = self.transform(pil_image)

        # Get annotations for this image
        annotations = self.img_to_anns.get(img_id, [])

        objects = []
        object_class_names = []
        object_class_ids = []

        # Scale factors
        scale_w = self.resolution / orig_w
        scale_h = self.resolution / orig_h

        for i, ann in enumerate(annotations):
            # Get bbox - format is [x, y, width, height] in COCO format
            bbox_coco = ann.get("bbox", None)
            if bbox_coco is None:
                continue

            # Get class name from category_id
            category_id = ann.get("category_id", 0)
            class_name = self.categories.get(category_id, "object")
            object_class_names.append(class_name)
            object_class_ids.append(int(category_id))

            # Convert from COCO [x, y, w, h] to [x1, y1, x2, y2]
            x, y, w, h = bbox_coco
            box_tensor = torch.tensor([x, y, x + w, y + h], dtype=torch.float32)

            # Scale box to resolution
            box_tensor[0] *= scale_w
            box_tensor[2] *= scale_w
            box_tensor[1] *= scale_h
            box_tensor[3] *= scale_h

            # IMPORTANT: Normalize boxes to [0, 1] range (required by SAM3 loss functions)
            box_tensor /= self.resolution

            # Handle segmentation mask (polygon or RLE format)
            segment = None
            segmentation = ann.get("segmentation", None)

            if segmentation:
                try:
                    # Check if it's RLE format (dict) or polygon format (list)
                    if isinstance(segmentation, dict):
                        # RLE format: {"counts": "...", "size": [h, w]}
                        mask_np = mask_utils.decode(segmentation)
                    elif isinstance(segmentation, list):
                        # Polygon format: [[x1, y1, x2, y2, ...], ...]
                        # Convert polygon to RLE, then decode
                        rles = mask_utils.frPyObjects(segmentation, orig_h, orig_w)
                        rle = mask_utils.merge(rles)
                        mask_np = mask_utils.decode(rle)
                    else:
                        print(f"Warning: Unknown segmentation format: {type(segmentation)}")
                        segment = None
                        continue

                    # Resize mask to model resolution
                    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
                    mask_t = torch.nn.functional.interpolate(
                        mask_t,
                        size=(self.resolution, self.resolution),
                        mode="nearest"
                    )
                    segment = mask_t.squeeze() > 0.5  # [1008, 1008] boolean tensor

                except Exception as e:
                    print(f"Warning: Error processing mask for image {img_id}, ann {i}: {e}")
                    segment = None

            obj = Object(
                bbox=box_tensor,
                area=(box_tensor[2]-box_tensor[0])*(box_tensor[3]-box_tensor[1]),
                object_id=i,
                segment=segment
            )
            objects.append(obj)

        image_obj = Image(
            data=image_tensor,
            objects=objects,
            size=(self.resolution, self.resolution)
        )

        # Construct Queries - one per prompt text (= one per class). Each query
        # maps to only the objects of its class and carries the REAL category
        # id in its inference metadata, so predictions can be labelled with the
        # class their prompt asked for (classification-aware scoring).
        from collections import defaultdict

        class_to_object_ids = defaultdict(list)
        prompt_to_category_id = {}
        for obj, class_name, category_id in zip(objects, object_class_names,
                                                object_class_ids):
            key = class_name.lower()
            class_to_object_ids[key].append(obj.object_id)
            prompt_to_category_id[key] = int(category_id)

        # Prompts evaluated on this image: the ones actually present, plus (in
        # classification mode) every class of the split as a negative prompt so
        # the presence head is scored on absent classes too.
        prompt_specs = [(prompt, ids) for prompt, ids in class_to_object_ids.items()]
        if self.all_class_queries:
            for prompt_text, category_id in self.eval_prompt_categories:
                if prompt_text in class_to_object_ids:
                    continue
                prompt_specs.append((prompt_text, []))
                prompt_to_category_id[prompt_text] = int(category_id)

        if not prompt_specs:
            prompt_specs = [("object", [])]
            prompt_to_category_id["object"] = -1

        queries = [
            FindQueryLoaded(
                query_text=query_text,
                image_id=0,
                object_ids_output=obj_ids,
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    # coco_image_id keeps the real COCO id; original_image_id is
                    # the INDEX into this dataset, which is what the GT builders
                    # use to fetch the source datapoint.
                    coco_image_id=img_id,
                    original_image_id=idx,
                    original_category_id=prompt_to_category_id.get(query_text, -1),
                    original_size=(orig_h, orig_w),
                    object_id=-1,
                    frame_index=-1
                )
            )
            for query_text, obj_ids in prompt_specs
        ]

        return Datapoint(
            find_queries=queries,
            images=[image_obj],
            raw_images=[pil_image]
        )


def _dilate_masks(binary_masks, radius):
    """Dilate binary masks [N, H, W] by `radius` pixels (max-pool dilation)."""
    if radius <= 0:
        return binary_masks
    kernel = 2 * radius + 1
    dilated = torch.nn.functional.max_pool2d(
        binary_masks.float().unsqueeze(1),
        kernel_size=kernel, stride=1, padding=radius
    ).squeeze(1)
    return dilated > 0.5


def merge_overlapping_masks(binary_masks, scores, boxes, iou_threshold=0.15, dilate=0):
    """
    Merge overlapping masks that likely represent the same object (e.g., crack segments).

    This is more aggressive than NMS - it MERGES masks instead of suppressing them.
    Useful for cracks where model splits one crack into many segments.

    Args:
        binary_masks: Binary masks [N, H, W]
        scores: Confidence scores [N]
        boxes: Bounding boxes [N, 4]
        iou_threshold: IoU threshold for merging (default: 0.15, lower = more aggressive)
        dilate: Dilation radius in pixels. When > 0, masks are dilated by this
            many pixels and any two fragments whose dilated masks TOUCH are
            merged. This joins disjoint portions of one crack (which have
            mask IoU 0 and are never merged by the IoU test alone). The
            merged output keeps the ORIGINAL (undilated) masks.

    Returns:
        Tuple of (merged_masks, merged_scores, merged_boxes)
    """
    if len(binary_masks) == 0:
        return binary_masks, scores, boxes

    # Sort by score (highest first)
    sorted_indices = torch.argsort(scores, descending=True)
    binary_masks = binary_masks[sorted_indices]
    scores = scores[sorted_indices]
    boxes = boxes[sorted_indices]

    dilated_masks = _dilate_masks(binary_masks, dilate) if dilate > 0 else None

    merged_masks = []
    merged_scores = []
    merged_boxes = []
    used = torch.zeros(len(binary_masks), dtype=torch.bool)

    for i in range(len(binary_masks)):
        if used[i]:
            continue

        current_mask = binary_masks[i].clone()
        current_dilated = dilated_masks[i].clone() if dilated_masks is not None else None
        current_score = scores[i].item()
        current_box = boxes[i]
        used[i] = True

        # Find overlapping masks and merge them
        for j in range(i + 1, len(binary_masks)):
            if used[j]:
                continue

            if current_dilated is not None:
                # Proximity criterion: merge if the dilated masks touch at all.
                # (IoU of thin dilated fragments is tiny even when they belong
                # to the same crack, so a threshold test would never fire.)
                should_merge = (current_dilated & dilated_masks[j]).any().item()
            else:
                # Overlap criterion: IoU of the raw masks
                intersection = (current_mask & binary_masks[j]).sum().item()
                union = (current_mask | binary_masks[j]).sum().item()
                iou = intersection / union if union > 0 else 0
                should_merge = iou > iou_threshold

            if should_merge:
                current_mask = current_mask | binary_masks[j]
                if current_dilated is not None:
                    current_dilated = current_dilated | dilated_masks[j]
                current_score = max(current_score, scores[j].item())
                used[j] = True

        merged_masks.append(current_mask)
        merged_scores.append(current_score)
        merged_boxes.append(current_box)

    if len(merged_masks) > 0:
        merged_masks = torch.stack(merged_masks)
        merged_scores = torch.tensor(merged_scores, device=scores.device)
        merged_boxes = torch.stack(merged_boxes)
    else:
        merged_masks = binary_masks[:0]
        merged_scores = scores[:0]
        merged_boxes = boxes[:0]

    return merged_masks, merged_scores, merged_boxes


def apply_sam3_nms(pred_logits, pred_masks, pred_boxes, prob_threshold=0.3, nms_iou_threshold=0.7, max_detections=100):
    """
    Apply SAM3's standard NMS pipeline to filter predictions.

    Args:
        pred_logits: [N, 1] logits
        pred_masks: [N, H, W] mask logits
        pred_boxes: [N, 4] boxes in normalized format
        prob_threshold: Score threshold for filtering (default: 0.3, SAM3 uses 0.5)
        nms_iou_threshold: IoU threshold for NMS (default: 0.7, SAM3 uses 0.5-0.7)
        max_detections: Maximum detections to keep (default: 100)

    Returns:
        Tuple of (filtered_masks, filtered_scores, filtered_boxes)
    """
    if len(pred_logits) == 0:
        return pred_masks[:0], pred_logits[:0].squeeze(-1), pred_boxes[:0]

    # Convert logits to probabilities
    pred_probs = torch.sigmoid(pred_logits).squeeze(-1)  # [N]

    # Convert mask logits to binary masks (sigmoid + threshold)
    pred_masks_sigmoid = torch.sigmoid(pred_masks)  # [N, H, W]
    pred_masks_binary = pred_masks_sigmoid > 0.5  # [N, H, W]

    # Apply SAM3's NMS
    # nms_masks expects: pred_probs [N], pred_masks [N, H, W], prob_threshold, iou_threshold
    # Returns: keep mask [N] of booleans
    keep_mask = nms_masks(
        pred_probs=pred_probs,
        pred_masks=pred_masks_binary.float(),  # NMS expects float masks
        prob_threshold=prob_threshold,
        iou_threshold=nms_iou_threshold
    )

    # Filter predictions
    filtered_masks = pred_masks_sigmoid[keep_mask]  # Keep sigmoid masks for later
    filtered_scores = pred_probs[keep_mask]
    filtered_boxes = pred_boxes[keep_mask]

    # Top-K selection by score
    if max_detections > 0 and len(filtered_scores) > max_detections:
        top_k_scores, top_k_indices = torch.topk(filtered_scores, k=max_detections, largest=True)
        filtered_masks = filtered_masks[top_k_indices]
        filtered_scores = top_k_scores
        filtered_boxes = filtered_boxes[top_k_indices]

    return filtered_masks, filtered_scores, filtered_boxes


def convert_predictions_to_coco_format(predictions_list, image_ids, resolution=288,
                                       prob_threshold=0.3, nms_iou_threshold=0.7, max_detections=100,
                                       merge_cracks=False, merge_iou_threshold=0.15,
                                       merge_dilate=0, use_presence=True):
    """
    Convert model predictions to COCO format using SAM3's NMS pipeline.

    Args:
        predictions_list: List of predictions per image
        image_ids: List of image IDs
        resolution: Resolution for box scaling (default: 288)
        prob_threshold: Score threshold (default: 0.3, SAM3 uses 0.5)
        nms_iou_threshold: NMS IoU threshold (default: 0.7)
        max_detections: Max detections per image (default: 100)
        merge_cracks: If True, merge overlapping segments instead of NMS suppression (default: False)
        merge_iou_threshold: IoU threshold for merging (default: 0.15, lower = more aggressive)
        merge_dilate: Dilation radius (pixels at 288 resolution) for merging.
            When > 0, disjoint fragments whose dilated masks touch are merged
            into one instance (joins split cracks). Only used with merge_cracks.
        use_presence: Multiply per-query scores by the prompt-level presence
            score, like SAM3's official PostProcessImage. The image model does
            NOT fold presence into pred_logits (supervise_joint_box_scores is
            only set for the video model), so raw logits alone are uncalibrated
            for the zero-shot baseline: queries score high even when the
            presence head says the concept is absent, flooding the eval with
            false positives (baseline mAP collapses to ~0).
    """
    coco_predictions = []
    pred_id = 0

    if merge_cracks:
        print(f"\n[INFO] Converting {len(predictions_list)} predictions to COCO format...")
        print(f"[INFO] Using CRACK MERGING mode: prob_threshold={prob_threshold}, merge_iou={merge_iou_threshold}, max_dets={max_detections}")
        if merge_dilate > 0:
            print(f"[INFO] Proximity merging ENABLED: fragments within ~{2*merge_dilate}px (dilate={merge_dilate}) are joined into one instance")
        print(f"[INFO] This will MERGE overlapping crack segments instead of suppressing them")
    else:
        print(f"\n[INFO] Converting {len(predictions_list)} predictions to COCO format...")
        print(f"[INFO] Using SAM3 NMS: prob_threshold={prob_threshold}, nms_iou={nms_iou_threshold}, max_dets={max_detections}")
    print(f"[INFO] Presence scoring: {'ENABLED (score = sigmoid(logit) * sigmoid(presence))' if use_presence else 'DISABLED (raw logits)'}")

    for img_id, preds in tqdm(zip(image_ids, predictions_list), total=len(predictions_list), desc="Converting predictions"):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        logits = preds['pred_logits']  # [N, 1]
        boxes = preds['pred_boxes']    # [N, 4]
        masks = preds['pred_masks']    # [N, H, W]

        # Joint score like SAM3's PostProcessImage (use_presence=True): multiply
        # each query's probability by the prompt-level presence probability,
        # then convert back to a logit so the downstream NMS/threshold code
        # (which applies sigmoid itself) operates on the joint score.
        presence_logit = preds.get('presence_logit', None)
        if use_presence and presence_logit is not None:
            presence_prob = torch.sigmoid(presence_logit.float()).reshape(-1)[0]
            joint_probs = torch.sigmoid(logits.float()) * presence_prob
            logits = torch.logit(joint_probs.clamp(1e-6, 1 - 1e-6))

        if merge_cracks:
            # Step 1: Filter by score threshold
            pred_probs = torch.sigmoid(logits).squeeze(-1)  # [N]
            valid_mask = pred_probs > prob_threshold

            filtered_masks = masks[valid_mask]
            filtered_scores = pred_probs[valid_mask]
            filtered_boxes = boxes[valid_mask]

            if len(filtered_masks) > 0:
                # Step 2: Convert masks to binary
                pred_masks_sigmoid = torch.sigmoid(filtered_masks)
                pred_masks_binary = (pred_masks_sigmoid > 0.5)

                # Step 3: MERGE overlapping crack segments
                merged_masks, merged_scores, merged_boxes = merge_overlapping_masks(
                    pred_masks_binary.cpu(),
                    filtered_scores.cpu(),
                    filtered_boxes.cpu(),
                    iou_threshold=merge_iou_threshold,
                    dilate=merge_dilate
                )

                # Step 4: Top-K selection by score
                if max_detections > 0 and len(merged_scores) > max_detections:
                    top_k_scores, top_k_indices = torch.topk(merged_scores, k=max_detections, largest=True)
                    merged_masks = merged_masks[top_k_indices]
                    merged_scores = top_k_scores
                    merged_boxes = merged_boxes[top_k_indices]

                # Return merged results (already binary)
                filtered_masks = merged_masks.float()  # Already binary, just convert to float
                filtered_scores = merged_scores
                filtered_boxes = merged_boxes
            else:
                filtered_masks = torch.tensor([])
                filtered_scores = torch.tensor([])
                filtered_boxes = torch.tensor([])
        else:
            # Apply SAM3's NMS pipeline (standard suppression)
            filtered_masks, filtered_scores, filtered_boxes = apply_sam3_nms(
                pred_logits=logits,
                pred_masks=masks,
                pred_boxes=boxes,
                prob_threshold=prob_threshold,
                nms_iou_threshold=nms_iou_threshold,
                max_detections=max_detections
            )

        if len(filtered_masks) > 0:
            # Convert filtered masks to binary for RLE encoding
            binary_masks = (filtered_masks > 0.5).cpu()
            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, filtered_scores.cpu().tolist(), filtered_boxes.cpu().tolist())):
                cx, cy, w, h = box
                x = (cx - w/2) * resolution
                y = (cy - h/2) * resolution
                w = w * resolution
                h = h * resolution

                pred_dict = {
                    'image_id': int(img_id),
                    'category_id': max(1, int(preds.get('category_id', 1))),
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                }

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def compute_pixel_metrics(coco_gt_dict, coco_predictions):
    """
    Pixel-level (semantic) segmentation metrics per (image, prompt) unit.

    Instance matching (mAP/cgF1) requires one prediction to cover one GT
    instance at IoU >= threshold, which triple-penalizes a crack predicted as
    several disjoint fragments (GT missed + each fragment a false positive).
    Here we instead union all predicted masks and all GT masks per unit and
    compare pixels, so fragmentation does not matter — only coverage does.
    This is the standard metric in the crack-segmentation literature.

    Predictions are the post-NMS/post-merge set (already score-filtered), so
    the prob_threshold acts as the binarization threshold.

    Returns dict with:
        pixel_iou:       micro-averaged IoU (all TP/FP/FN pixels pooled)
        pixel_precision: micro-averaged precision
        pixel_recall:    micro-averaged recall
        mean_unit_iou:   mean per-unit IoU over units where GT or prediction
                         is non-empty (empty-vs-empty units are skipped)
    """
    from collections import defaultdict

    def _decode_rle(rle):
        rle = dict(rle)
        if isinstance(rle.get('counts'), str):
            rle['counts'] = rle['counts'].encode('utf-8')
        return mask_utils.decode(rle).astype(bool)

    preds_by_img = defaultdict(list)
    for p in coco_predictions:
        preds_by_img[p['image_id']].append(p['segmentation'])

    gts_by_img = defaultdict(list)
    for a in coco_gt_dict['annotations']:
        if 'segmentation' in a:
            gts_by_img[a['image_id']].append(a['segmentation'])

    tp = fp = fn = 0
    unit_ious = []

    for img in coco_gt_dict['images']:
        img_id = img['id']
        h, w = img['height'], img['width']

        gt_union = np.zeros((h, w), dtype=bool)
        for rle in gts_by_img.get(img_id, []):
            gt_union |= _decode_rle(rle)

        pred_union = np.zeros((h, w), dtype=bool)
        for rle in preds_by_img.get(img_id, []):
            pred_union |= _decode_rle(rle)

        inter = int(np.logical_and(gt_union, pred_union).sum())
        union = int(np.logical_or(gt_union, pred_union).sum())
        tp += inter
        fp += int(pred_union.sum()) - inter
        fn += int(gt_union.sum()) - inter
        if union > 0:
            unit_ious.append(inter / union)

    return {
        'pixel_iou': tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0,
        'pixel_precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'pixel_recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'mean_unit_iou': float(np.mean(unit_ious)) if unit_ious else 0.0,
    }


def create_coco_gt_from_dataset(dataset, image_ids=None, mask_resolution=288):
    """
    Create COCO ground truth dictionary from dataset.

    OPTIMIZATION: Downsample GT masks to 288×288 to match prediction resolution.
    """
    print(f"\n[INFO] Creating COCO ground truth (downsampling to {mask_resolution}×{mask_resolution})...")

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    for idx in tqdm(list(indices), desc="Creating GT"):
        coco_gt['images'].append({
            'id': int(idx),
            'width': mask_resolution,
            'height': mask_resolution,
            'is_instance_exhaustive': True
        })

        datapoint = dataset[idx]

        for obj in datapoint.images[0].objects:
            # Scale boxes to mask_resolution
            box = obj.bbox * mask_resolution
            x1, y1, x2, y2 = box.tolist()
            x, y, w, h = x1, y1, x2-x1, y2-y1

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                # Downsample mask from 1008×1008 to mask_resolution×mask_resolution
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                downsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(mask_resolution, mask_resolution),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = downsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    print(f"[INFO] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")

    return coco_gt


def create_coco_gt_per_prompt(dataset, eval_units, mask_resolution=288,
                              categories=None):
    """
    Create COCO ground truth where each evaluation unit is a single
    (image, text-prompt) pair rather than a whole image.

    This is the correct GT for SAM3 validation: the model produces one set of
    predictions per text prompt, and an image may carry several prompts (one per
    category). Each unit's GT therefore contains ONLY the objects that the
    prompt targets, so a prediction for "crack" is never scored against "hole"
    boxes. Empty/negative prompts produce a GT image with no annotations.

    Args:
        dataset: the validation dataset.
        eval_units: list of dicts, each with
            - 'eval_id': unique COCO image id for this (image, prompt) pair
            - 'src_idx': index into `dataset` of the source image
            - 'obj_ids': object ids (into datapoint.images[0].objects) that this
                         prompt is responsible for (empty list for negatives)
        mask_resolution: resolution GT masks/boxes are scaled to (matches preds).
        categories: COCO 'categories' list to publish. The REAL category ids are
            used, so class-agnostic (useCats=False) and per-class
            (useCats=True) COCOeval can both run on the same file.
    """
    print(f"\n[INFO] Creating per-prompt COCO ground truth "
          f"(downsampling to {mask_resolution}×{mask_resolution})...")

    categories = categories or [{'id': 1, 'name': 'object'}]
    valid_cat_ids = {int(c['id']) for c in categories}
    fallback_cat_id = int(categories[0]['id'])

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset (per-prompt)',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': categories
    }

    ann_id = 0
    # Cache decoded datapoints so we read/decode each source image only once
    # even when it contributes several prompts.
    datapoint_cache = {}

    for unit in tqdm(eval_units, desc="Creating GT"):
        eval_id = int(unit['eval_id'])
        src_idx = int(unit['src_idx'])
        obj_ids = set(int(o) for o in unit['obj_ids'])
        unit_cat_id = int(unit.get('category_id', -1))
        if unit_cat_id not in valid_cat_ids:
            unit_cat_id = fallback_cat_id

        coco_gt['images'].append({
            'id': eval_id,
            'width': mask_resolution,
            'height': mask_resolution,
            'is_instance_exhaustive': True
        })

        if src_idx not in datapoint_cache:
            datapoint_cache[src_idx] = dataset[src_idx]
        datapoint = datapoint_cache[src_idx]

        for obj in datapoint.images[0].objects:
            # Only the objects this prompt is responsible for.
            if obj.object_id not in obj_ids:
                continue

            # Scale boxes to mask_resolution (bboxes are normalized 0..1)
            box = obj.bbox * mask_resolution
            x1, y1, x2, y2 = box.tolist()
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

            ann = {
                'id': ann_id,
                'image_id': eval_id,
                'category_id': unit_cat_id,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                downsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(mask_resolution, mask_resolution),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = downsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    print(f"[INFO] Created {len(coco_gt['images'])} prompt units, "
          f"{len(coco_gt['annotations'])} annotations")

    return coco_gt


def create_coco_gt_per_image(dataset, src_indices, mask_resolution=288,
                             categories=None):
    """
    COCO ground truth with the REAL category ids, one entry per SOURCE image.

    This is the GT needed for classification-aware evaluation: predictions from
    every prompt of an image (one prompt per class) land in the same GT image,
    each labelled with the category id its prompt asked for. COCOeval with
    useCats=True then charges a correct-class detection as TP, and a detection
    produced by the WRONG class prompt as FP for that class + FN for the true
    one — i.e. misclassification is penalised instead of hidden.

    Args:
        dataset: validation dataset exposing `object_categories_by_index`.
        src_indices: dataset indices of the source images to include.
        mask_resolution: resolution GT boxes/masks are scaled to (matches preds).
        categories: COCO 'categories' list to publish.
    """
    print(f"\n[INFO] Creating per-image COCO ground truth with real categories "
          f"(downsampling to {mask_resolution}×{mask_resolution})...")

    categories = categories or [{'id': 1, 'name': 'object'}]
    valid_cat_ids = {int(c['id']) for c in categories}
    fallback_cat_id = int(categories[0]['id'])

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset (per-image, per-class)',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': categories
    }

    ann_id = 0
    for src_idx in tqdm(list(src_indices), desc="Creating GT (per image)"):
        src_idx = int(src_idx)
        datapoint = dataset[src_idx]
        cat_by_object_id = dataset.object_categories_by_index[src_idx]

        coco_gt['images'].append({
            'id': src_idx,
            'width': mask_resolution,
            'height': mask_resolution,
            'is_instance_exhaustive': True
        })

        for obj in datapoint.images[0].objects:
            cat_id = int(cat_by_object_id.get(obj.object_id, -1))
            if cat_id not in valid_cat_ids:
                cat_id = fallback_cat_id

            box = obj.bbox * mask_resolution
            x1, y1, x2, y2 = box.tolist()
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

            ann = {
                'id': ann_id,
                'image_id': src_idx,
                'category_id': cat_id,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                downsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(mask_resolution, mask_resolution),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = downsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    print(f"[INFO] Created {len(coco_gt['images'])} images, "
          f"{len(coco_gt['annotations'])} annotations (real category ids)")

    return coco_gt


def _pred_present(preds, threshold=0.5):
    """
    Prompt-level "is this class present in the image?" decision.

    Uses the presence head (SAM3's official PostProcessImage does the same).
    Falls back to the best query probability when the presence head is absent.
    """
    if preds is None:
        return None
    presence = preds.get('presence_logit', None)
    if presence is not None:
        return bool(torch.sigmoid(presence.float()).reshape(-1)[0].item() > threshold)
    logits = preds.get('pred_logits', None)
    if logits is None or len(logits) == 0:
        return False
    return bool(torch.sigmoid(logits.float()).max().item() > threshold)


def compute_prompt_classification_metrics(eval_units, predictions,
                                          cat_id_to_name, threshold=0.5):
    """
    Prompt-level classification metrics ("does the model say this class is
    here?"), one decision per (image, prompt) pair — this is the classification
    signal the detection metrics cannot express for absent classes.

    Returns dict with micro accuracy/precision/recall/F1 plus per-class rows.
    """
    per_class = {}
    correct = total = 0
    skipped = 0

    for unit, preds in zip(eval_units, predictions):
        cat_id = int(unit.get('category_id', -1))
        if cat_id <= 0:
            skipped += 1
            continue
        pred_present = _pred_present(preds, threshold)
        if pred_present is None:
            skipped += 1
            continue

        gt_present = len(unit['obj_ids']) > 0
        row = per_class.setdefault(
            cat_id,
            {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'pos': 0, 'neg': 0}
        )
        row['pos' if gt_present else 'neg'] += 1
        if gt_present and pred_present:
            row['tp'] += 1
        elif gt_present and not pred_present:
            row['fn'] += 1
        elif not gt_present and pred_present:
            row['fp'] += 1
        else:
            row['tn'] += 1
        total += 1
        correct += int(gt_present == pred_present)

    def _prf(row):
        tp, fp, fn = row['tp'], row['fp'], row['fn']
        n = tp + fp + fn
        acc = (tp + row['tn']) / n if n > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return acc, prec, rec, f1

    rows = {}
    for cat_id, row in per_class.items():
        acc, prec, rec, f1 = _prf(row)
        rows[cat_id] = {
            'name': cat_id_to_name.get(cat_id, str(cat_id)),
            'pos': row['pos'], 'neg': row['neg'],
            'tp': row['tp'], 'fp': row['fp'],
            'fn': row['fn'], 'tn': row['tn'],
            'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1,
        }

    return {
        'threshold': threshold,
        'num_units': total,
        'num_skipped': skipped,
        'accuracy': correct / total if total > 0 else 0.0,
        'macro_precision': float(np.mean([r['precision'] for r in rows.values()])) if rows else 0.0,
        'macro_recall': float(np.mean([r['recall'] for r in rows.values()])) if rows else 0.0,
        'macro_f1': float(np.mean([r['f1'] for r in rows.values()])) if rows else 0.0,
        'per_class': rows,
    }


def print_per_class_ap(coco_eval, cat_id_to_name):
    """
    Print per-class AP from a COCOeval run with useCats=True.

    COCOeval precision tensor is [T, R, K, A, M]; area index 0 is 'all' and
    maxDet index 2 is 100 — the same slices pycocotools' summarize() uses.
    """
    precisions = coco_eval.eval['precision']
    if precisions.size == 0:
        print("[WARN] Per-class evaluation produced no precision values")
        return {}

    cat_ids = coco_eval.params.catIds
    rows = {}
    for k, cat_id in enumerate(cat_ids):
        p_all = precisions[:, :, k, 0, 2]
        p_all = p_all[p_all > -1]
        if p_all.size == 0:
            continue
        p50 = precisions[0, :, k, 0, 2]
        p50 = p50[p50 > -1]
        rows[int(cat_id)] = {
            'name': cat_id_to_name.get(int(cat_id), str(cat_id)),
            'ap': float(p_all.mean()),
            'ap50': float(p50.mean()) if p50.size else 0.0,
        }

    if rows:
        print(f"{'class':<24}{'AP':>8}{'AP50':>8}")
        for r in rows.values():
            print(f"{r['name'][:23]:<24}{r['ap']:>8.4f}{r['ap50']:>8.4f}")
        print(f"{'MACRO':<24}"
              f"{np.mean([r['ap'] for r in rows.values()]):>8.4f}"
              f"{np.mean([r['ap50'] for r in rows.values()]):>8.4f}")
    return rows


def build_confusion_matrix(predictions, gt_dict, cat_id_to_name,
                           iou_threshold=0.5, resolution=288,
                           use_boxes=True):
    """
    Detection confusion matrix (rows = predicted class, cols = true class) with
    an extra 'background' row/column, in the style of ultralytics'
    ConfusionMatrix.

    Instances are matched per image by a greedy assignment at `iou_threshold`,
    using BOX IoU by default (`use_boxes=True`) — mask IoU is quantised away for
    thin objects such as a ~2-px needle mask at 288 resolution. A matched pair
    is charged to (predicted class, GT class),
    so a correct-class hit lands on the diagonal and a wrong-class hit lands
    off-diagonal. Unmatched GT -> (background, GT class); unmatched prediction
    -> (predicted class, background).

    Returns:
        (matrix, names): np.ndarray [nc+1, nc+1] and the tick labels.
    """
    from collections import defaultdict

    cat_ids = sorted(cat_id_to_name)
    idx_of = {cid: i for i, cid in enumerate(cat_ids)}
    nc = len(cat_ids)
    bg = nc
    matrix = np.zeros((nc + 1, nc + 1), dtype=np.float64)

    def _mask(item):
        seg = item.get('segmentation')
        if seg is not None:
            return mask_utils.decode(seg).astype(bool)
        x, y, w, h = item['bbox']
        m = np.zeros((resolution, resolution), dtype=bool)
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(resolution, int(x + w)), min(resolution, int(y + h))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = True
        return m

    preds_by_img = defaultdict(list)
    for p in predictions:
        preds_by_img[int(p['image_id'])].append(p)

    gts_by_img = defaultdict(list)
    for a in gt_dict['annotations']:
        gts_by_img[int(a['image_id'])].append(a)

    for img in gt_dict['images']:
        img_id = int(img['id'])
        preds = preds_by_img.get(img_id, [])
        gts = gts_by_img.get(img_id, [])

        if not gts:
            for p in preds:
                matrix[idx_of.get(int(p['category_id']), bg), bg] += 1
            continue
        if not preds:
            for g in gts:
                matrix[bg, idx_of.get(int(g['category_id']), bg)] += 1
            continue

        gt_masks = None if use_boxes else [_mask(g) for g in gts]
        pred_masks = None if use_boxes else [_mask(p) for p in preds]

        iou = np.zeros((len(gts), len(preds)), dtype=np.float64)
        if use_boxes:
            # Box IoU: far more forgiving than mask IoU for thin objects such as
            # a ~2-px needle mask at 288 resolution.
            for i, g in enumerate(gts):
                gx, gy, gw, gh = g['bbox']
                for j, p in enumerate(preds):
                    px, py, pw, ph = p['bbox']
                    iw = max(0.0, min(gx + gw, px + pw) - max(gx, px))
                    ih = max(0.0, min(gy + gh, py + ph) - max(gy, py))
                    inter = iw * ih
                    union = gw * gh + pw * ph - inter
                    iou[i, j] = inter / union if union > 0 else 0.0
        else:
            for i, gm in enumerate(gt_masks):
                gm_sum = gm.sum()
                for j, pm in enumerate(pred_masks):
                    inter = np.logical_and(gm, pm).sum()
                    union = gm_sum + pm.sum() - inter
                    iou[i, j] = inter / union if union > 0 else 0.0

        gi, pj = np.where(iou >= iou_threshold)
        pairs = sorted(zip(iou[gi, pj], gi, pj), key=lambda t: -t[0])
        used_g, used_p = set(), set()
        for _, i, j in pairs:
            i, j = int(i), int(j)
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            gc = idx_of.get(int(gts[i]['category_id']), bg)
            pc = idx_of.get(int(preds[j]['category_id']), bg)
            matrix[pc, gc] += 1

        for i, g in enumerate(gts):
            if i not in used_g:
                matrix[bg, idx_of.get(int(g['category_id']), bg)] += 1
        for j, p in enumerate(preds):
            if j not in used_p:
                matrix[idx_of.get(int(p['category_id']), bg), bg] += 1

    names = [cat_id_to_name[c] for c in cat_ids] + ['background']
    return matrix, names


def plot_confusion_matrix(matrix, names, save_dir, normalize=True,
                          title="confusion_matrix"):
    """Render the confusion matrix as a heatmap PNG (matplotlib only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    arr = matrix / (matrix.sum(axis=0, keepdims=True) + 1e-9) if normalize else matrix
    disp = np.nan_to_num(arr, nan=0.0)

    n = len(names)
    fig, ax = plt.subplots(figsize=(max(6.0, n * 1.2), max(5.0, n * 1.0)))
    vmax = 1.0 if normalize else max(disp.max(), 1e-9)
    im = ax.imshow(disp, cmap="Blues", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))

    threshold = 0.005 if normalize else 0.5
    for i in range(n):
        for j in range(n):
            v = arr[i, j]
            if np.isnan(v) or v < threshold:
                continue
            ax.text(j, i, f"{v:.2f}" if normalize else f"{v:.0f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if v > 0.5 * vmax else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fname = save_dir / f"{title}{'_normalized' if normalize else ''}.png"
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    return fname


def plot_bar_chart(labels, values, ylabel, title, save_dir, fname,
                   ylim=None, color="#4C72B0"):
    """Single-series bar chart PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(6.0, len(labels) * 0.9), 5.0))
    bars = ax.bar(range(len(labels)), values, color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = save_dir / fname
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_grouped_bars(labels, series, ylabel, title, save_dir, fname,
                      ylim=(0, 1)):
    """Grouped bar chart (e.g. precision / recall / F1 per class)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n = len(labels)
    k = len(series)
    width = 0.8 / max(k, 1)
    fig, ax = plt.subplots(figsize=(max(7.0, n * 1.2), 5.0))
    xs = np.arange(n)
    for s_idx, (name, values) in enumerate(series.items()):
        ax.bar(xs + (s_idx - (k - 1) / 2) * width, values, width,
               label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = save_dir / fname
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_pr_curves(coco_eval, cat_id_to_name, save_dir, iou_index=0,
                   fname="pr_curve.png"):
    """
    Precision-recall curve from a COCOeval precision tensor [T, R, K, A, M].

    `iou_index=0` selects IoU 0.50; area index 0 is 'all', maxDet index 2 is 100.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision = coco_eval.eval.get('precision') if coco_eval is not None else None
    if precision is None or precision.size == 0:
        return None

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    recall_grid = np.asarray(coco_eval.params.recThrs)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    curves = []
    for k, cat_id in enumerate(coco_eval.params.catIds):
        prec = precision[iou_index, :, k, 0, 2]
        valid = prec > -1
        if valid.sum() == 0:
            continue
        name = cat_id_to_name.get(int(cat_id), str(cat_id))
        ax.plot(recall_grid[valid], prec[valid], linewidth=1, label=name)
        curves.append(np.where(valid, prec, np.nan))

    if curves:
        mean_curve = np.nanmean(np.vstack(curves), axis=0)
        ax.plot(recall_grid, mean_curve, linewidth=3, color="blue",
                label="all classes")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve @ IoU 0.50 (classification-aware)")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = save_dir / fname
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_validation_plots(save_dir, confusion_matrix=None, confusion_names=None,
                          per_class_ap=None, cls_metrics=None, coco_eval=None,
                          cat_id_to_name=None, summary=None):
    """Write every validation plot into `save_dir`; returns the file paths."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    written = []

    if confusion_matrix is not None and confusion_names is not None:
        written.append(plot_confusion_matrix(
            confusion_matrix, confusion_names, save_dir, normalize=False))
        written.append(plot_confusion_matrix(
            confusion_matrix, confusion_names, save_dir, normalize=True))

    if per_class_ap:
        names = [r['name'] for r in per_class_ap.values()]
        written.append(plot_bar_chart(
            names, [r['ap'] for r in per_class_ap.values()],
            "AP (IoU 0.50:0.95)", "Per-class AP (classification-aware)",
            save_dir, "per_class_ap.png", ylim=(0, 1)))
        written.append(plot_bar_chart(
            names, [r['ap50'] for r in per_class_ap.values()],
            "AP@50", "Per-class AP@50 (classification-aware)",
            save_dir, "per_class_ap50.png", ylim=(0, 1), color="#DD8452"))

    if cls_metrics and cls_metrics.get('per_class'):
        rows = list(cls_metrics['per_class'].values())
        names = [r['name'] for r in rows]
        written.append(plot_grouped_bars(
            names,
            {
                'precision': [r['precision'] for r in rows],
                'recall': [r['recall'] for r in rows],
                'f1': [r['f1'] for r in rows],
            },
            "Score", "Prompt-level classification (presence head)",
            save_dir, "prompt_classification_prf.png"))
        written.append(plot_bar_chart(
            names, [r['accuracy'] for r in rows],
            "Accuracy", "Prompt-level classification accuracy",
            save_dir, "prompt_classification_accuracy.png",
            ylim=(0, 1), color="#55A868"))

    if coco_eval is not None:
        pr_path = plot_pr_curves(coco_eval, cat_id_to_name or {}, save_dir)
        if pr_path is not None:
            written.append(pr_path)

    if summary:
        written.append(plot_bar_chart(
            list(summary.keys()), list(summary.values()),
            "Score", "Validation metrics", save_dir, "metrics_summary.png",
            ylim=(0, 1), color="#8172B3"))

    return written


def save_class_overlays(dataset, predictions, gt_dict, classes, out_dir,
                        iou_threshold=0.5, resolution=288, max_images=None):
    """
    Draw GT + predicted boxes per target class and save one PNG per image.

    Colour code:
        green  -> TP  (prediction matched to a same-class GT at IoU >= thr)
        red    -> FN  (GT with no matching prediction; label carries class name)
        yellow -> FP  (prediction with no matching GT; label carries class name)

    Args:
        dataset: validation dataset (used for original image paths/sizes).
        predictions: COCO predictions with real category ids (288-res coords).
        gt_dict: COCO GT with real category ids (288-res coords, per image).
        classes: iterable of class names to render (case-insensitive).
        out_dir: root directory; each class gets its own subfolder.
        iou_threshold: box IoU used to match prediction <-> GT.
        resolution: coordinate space of the COCO boxes (288 by default).
        max_images: optional cap per class (debug).

    Returns:
        dict class name -> {'tp', 'fp', 'fn', 'images'}
    """
    from collections import defaultdict
    from PIL import ImageDraw, ImageFont

    out_dir = Path(out_dir)
    name_to_id = {str(n).lower(): int(c) for c, n in dataset.categories.items()}
    targets = []
    for c in classes:
        cid = name_to_id.get(str(c).lower())
        if cid is None:
            print(f"[WARN] Unknown overlay class '{c}' — skipped")
            continue
        targets.append((str(c).lower(), cid))
    if not targets:
        return {}

    def _iou(a, b):
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        ih = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    preds_by_img = defaultdict(list)
    for p in predictions:
        preds_by_img[int(p['image_id'])].append(p)

    gts_by_img = defaultdict(list)
    for a in gt_dict['annotations']:
        gts_by_img[int(a['image_id'])].append(a)

    stats = {name: {'tp': 0, 'fp': 0, 'fn': 0, 'images': 0} for name, _ in targets}
    color_tp = (0, 170, 60)
    color_fn = (220, 20, 20)
    color_fp = (235, 200, 0)

    for name, cid in targets:
        class_dir = out_dir / name
        class_dir.mkdir(parents=True, exist_ok=True)
        n_saved = 0

        for img in gt_dict['images']:
            src_idx = int(img['id'])
            preds = [p for p in preds_by_img.get(src_idx, [])
                     if int(p['category_id']) == cid]
            gts = [g for g in gts_by_img.get(src_idx, [])
                   if int(g['category_id']) == cid]
            if not preds and not gts:
                continue

            # Greedy match predictions to GT by box IoU.
            pairs = sorted(
                ((_iou(p['bbox'], g['bbox']), pi, gi)
                 for pi, p in enumerate(preds)
                 for gi, g in enumerate(gts)),
                key=lambda t: -t[0]
            )
            matched_pred, matched_gt = {}, {}
            for iou, pi, gi in pairs:
                if iou < iou_threshold or pi in matched_pred or gi in matched_gt:
                    continue
                matched_pred[pi] = gi
                matched_gt[gi] = pi

            img_info = dataset.images[dataset.image_ids[src_idx]]
            img_path = dataset.split_dir / img_info['file_name']
            try:
                pil_img = PILImage.open(img_path).convert("RGB")
            except Exception as e:
                print(f"[WARN] Cannot open {img_path}: {e}")
                continue

            ow, oh = pil_img.size
            sx, sy = ow / resolution, oh / resolution

            def _scale(box):
                x, y, w, h = box
                return [x * sx, y * sy, (x + w) * sx, (y + h) * sy]

            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None

            def _label(box, text, color):
                x1, y1, _, _ = box
                ty = max(0, y1 - 12)
                draw.text((x1 + 2, ty), text, fill=color, font=font)

            for gi, g in enumerate(gts):
                box = _scale(g['bbox'])
                if gi in matched_gt:
                    draw.rectangle(box, outline=color_tp, width=3)
                    _label(box, f"{name} TP", color_tp)
                    stats[name]['tp'] += 1
                else:
                    draw.rectangle(box, outline=color_fn, width=3)
                    _label(box, f"{name} FN", color_fn)
                    stats[name]['fn'] += 1

            for pi, p in enumerate(preds):
                if pi in matched_pred:
                    continue
                box = _scale(p['bbox'])
                draw.rectangle(box, outline=color_fp, width=3)
                _label(box, f"{name} FP", color_fp)
                stats[name]['fp'] += 1

            out_path = class_dir / f"{Path(img_info['file_name']).stem}.png"
            pil_img.save(out_path)
            stats[name]['images'] += 1
            n_saved += 1
            if max_images and n_saved >= max_images:
                break

        print(f"[INFO] Overlays '{name}': {n_saved} images -> {class_dir} "
              f"(TP={stats[name]['tp']} FP={stats[name]['fp']} "
              f"FN={stats[name]['fn']})")

    print(f"[INFO] Overlay colour code: green=TP, red=FN (GT), yellow=FP (pred)")

    return stats


def convert_predictions_to_coco_format_original_res(predictions_list, image_ids, dataset, model_resolution=288, score_threshold=0.0, merge_overlaps=True, iou_threshold=0.3, debug=False):
    """
    Convert model predictions to COCO format at ORIGINAL image resolution.

    This matches the inference approach (infer_sam.py) where:
    1. Masks are upsampled from 288x288 to original image size
    2. Boxes are scaled to original image size
    3. Evaluation happens at original resolution

    Args:
        predictions_list: List of predictions per image
        image_ids: List of image IDs (indices into dataset)
        dataset: Dataset to get original image sizes
        model_resolution: Model output resolution (default: 288)
        score_threshold: Confidence threshold
        merge_overlaps: Whether to merge overlapping predictions
        iou_threshold: IoU threshold for merging
        debug: Print debug info
    """
    coco_predictions = []
    pred_id = 0

    if debug:
        print(f"\n[DEBUG] Converting {len(predictions_list)} predictions to COCO format (ORIGINAL RESOLUTION)...")
        if merge_overlaps:
            print(f"[DEBUG] Overlapping segment merging ENABLED (IoU threshold={iou_threshold})")

    for img_id, preds in zip(image_ids, predictions_list):
        if preds is None or len(preds.get('pred_logits', [])) == 0:
            continue

        # Get original image size from dataset
        datapoint = dataset[img_id]
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        logits = preds['pred_logits']
        boxes = preds['pred_boxes']
        masks = preds['pred_masks']  # [N, 288, 288]

        scores = torch.sigmoid(logits).squeeze(-1)

        # Filter by score threshold
        valid_mask = scores > score_threshold
        num_before = len(scores)
        scores = scores[valid_mask]
        boxes = boxes[valid_mask]
        masks = masks[valid_mask]

        if debug and img_id == image_ids[0]:
            print(f"[DEBUG] Image {img_id}: {num_before} queries -> {len(scores)} after filtering (threshold={score_threshold})")
            if len(scores) > 0:
                print(f"[DEBUG]   Original size: {orig_w}x{orig_h}")
                print(f"[DEBUG]   Filtered scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        if len(masks) == 0:
            continue

        # Upsample masks from 288x288 to original resolution (like infer_sam.py)
        masks_sigmoid = torch.sigmoid(masks)  # [N, 288, 288]
        masks_upsampled = torch.nn.functional.interpolate(
            masks_sigmoid.unsqueeze(1).float(),  # [N, 1, 288, 288]
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [N, orig_h, orig_w]

        binary_masks = (masks_upsampled > 0.5).cpu()

        # Merge overlapping predictions
        if merge_overlaps and len(binary_masks) > 0:
            num_before_merge = len(binary_masks)
            binary_masks, scores, boxes = merge_overlapping_masks(
                binary_masks, scores.cpu(), boxes.cpu(), iou_threshold=iou_threshold
            )
            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Merged {num_before_merge} predictions -> {len(binary_masks)} (IoU threshold={iou_threshold})")

        if len(binary_masks) > 0:
            mask_areas = binary_masks.flatten(1).sum(1)

            if debug and img_id == image_ids[0]:
                print(f"[DEBUG]   Upsampled mask shape: {binary_masks.shape}")
                print(f"[DEBUG]   Mask areas: min={mask_areas.min():.0f}, max={mask_areas.max():.0f}, mean={mask_areas.float().mean():.0f}")

            rles = rle_encode(binary_masks)

            for idx, (rle, score, box) in enumerate(zip(rles, scores.cpu().tolist(), boxes.cpu().tolist())):
                # Convert box from normalized [0,1] to original image coordinates
                cx, cy, w_norm, h_norm = box
                x = (cx - w_norm/2) * orig_w
                y = (cy - h_norm/2) * orig_h
                w = w_norm * orig_w
                h = h_norm * orig_h

                # Clamp coordinates to image bounds
                x = max(0, min(x, orig_w))
                y = max(0, min(y, orig_h))
                w = max(0, min(w, orig_w - x))
                h = max(0, min(h, orig_h - y))

                # Skip if box is too small after clamping
                if w < 1 or h < 1:
                    continue

                pred_dict = {
                    'image_id': int(img_id),
                    'category_id': 1,
                    'segmentation': rle,
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                    'id': pred_id
                }

                if debug and img_id == image_ids[0] and idx == 0:
                    print(f"[DEBUG]   First prediction bbox (at {orig_w}x{orig_h}): {pred_dict['bbox']}")

                coco_predictions.append(pred_dict)
                pred_id += 1

    return coco_predictions


def create_coco_gt_from_dataset_original_res(dataset, image_ids=None, debug=False):
    """
    Create COCO ground truth dictionary from dataset at ORIGINAL resolution.

    This matches the inference approach (infer_sam.py) where GT is kept
    at original image size for evaluation.

    Args:
        dataset: Dataset with images and annotations
        image_ids: List of image IDs to include (None = all)
        debug: Print debug info
    """
    if debug:
        print(f"\n[DEBUG] Creating COCO ground truth (ORIGINAL RESOLUTION)...")

    coco_gt = {
        'info': {
            'description': 'SAM3 LoRA Validation Dataset',
            'version': '1.0',
            'year': 2024
        },
        'images': [],
        'annotations': [],
        'categories': [{'id': 1, 'name': 'object'}]
    }

    ann_id = 0
    indices = range(len(dataset)) if image_ids is None else image_ids

    for idx in indices:
        datapoint = dataset[idx]

        # Get original image size
        orig_h, orig_w = datapoint.find_queries[0].inference_metadata.original_size

        coco_gt['images'].append({
            'id': int(idx),
            'width': orig_w,
            'height': orig_h,
            'is_instance_exhaustive': True
        })

        for obj in datapoint.images[0].objects:
            # Scale boxes from normalized [0,1] to original size
            box = obj.bbox  # Already in [0,1] normalized coordinates
            x1, y1, x2, y2 = box.tolist()

            # Convert to original image coordinates
            x1_orig = x1 * orig_w
            y1_orig = y1 * orig_h
            x2_orig = x2 * orig_w
            y2_orig = y2 * orig_h

            # Convert to COCO format [x, y, w, h]
            x, y = x1_orig, y1_orig
            w = x2_orig - x1_orig
            h = y2_orig - y1_orig

            ann = {
                'id': ann_id,
                'image_id': int(idx),
                'category_id': 1,
                'bbox': [x, y, w, h],
                'area': w * h,
                'iscrowd': 0,
                'ignore': 0
            }

            if obj.segment is not None:
                # Upsample mask from 1008x1008 to original size
                mask_tensor = obj.segment.unsqueeze(0).unsqueeze(0).float()
                upsampled_mask = torch.nn.functional.interpolate(
                    mask_tensor,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                ) > 0.5

                mask_np = upsampled_mask.squeeze().cpu().numpy().astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask_np))
                rle['counts'] = rle['counts'].decode('utf-8')
                ann['segmentation'] = rle

            coco_gt['annotations'].append(ann)
            ann_id += 1

    if debug:
        print(f"[DEBUG] Created {len(coco_gt['images'])} images, {len(coco_gt['annotations'])} annotations")
        if len(coco_gt['annotations']) > 0:
            sample_gt = coco_gt['annotations'][0]
            sample_img = coco_gt['images'][0]
            print(f"[DEBUG] Sample GT: image_id={sample_gt['image_id']}, bbox={sample_gt['bbox']}, image_size={sample_img['width']}x{sample_img['height']}")

    return coco_gt


def move_to_device(obj, device):
    """Recursively move objects to device"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            val = getattr(obj, field)
            setattr(obj, field, move_to_device(val, device))
        return obj
    return obj


def validate(config_path, weights_path, val_data_dir, num_samples=None,
             prob_threshold=0.3, nms_iou=0.7, merge_cracks=False, merge_iou=0.15,
             merge_dilate=0, use_base_model=False, use_presence=True,
             pixel_metrics=True, classification=True,
             classification_threshold=0.5, plots=True, plots_dir=None,
             cm_iou=0.5, overlays=True, overlay_classes=None,
             overlay_dir=None, overlay_iou=0.5, cm_boxes=True):
    """Run validation with full metrics (mAP, cgF1) and SAM3 NMS

    Args:
        config_path: Path to config file (for LoRA settings only). Not required if use_base_model=True.
        weights_path: Path to LoRA weights. Not required if use_base_model=True.
        val_data_dir: Direct path to validation data directory containing _annotations.coco.json
                      (e.g., /workspace/data2/valid)
        num_samples: Optional limit for number of samples (for debugging)
        use_base_model: If True, use original SAM3 model without LoRA (default: False)
        classification: If True (default), query EVERY class on every image and
            report prompt-level classification metrics plus per-class AP with
            the real category ids. Set False for the old class-agnostic-only
            behaviour.
        classification_threshold: Presence probability above which a class is
            declared present (default: 0.5).
        plots: If True (default), save confusion matrix and metric plots as
            PNGs into `plots_dir`.
        plots_dir: Output directory for plots (default: outputs/validation_plots).
        cm_iou: Mask/box IoU threshold used to match predictions to GT when
            building the confusion matrix (default: 0.5).
        cm_boxes: If True (default), the confusion matrix matches on BOX IoU;
            set False to match on mask IoU.
        overlays: If True (default), save GT/prediction overlay images per
            target class (green=TP, red=FN, yellow=FP).
        overlay_classes: Class names to draw (default: all split classes).
        overlay_dir: Output root for overlays (default: <plots_dir>/detections).
        overlay_iou: Box IoU used to match prediction <-> GT in the overlays.

    Example (with LoRA):
        validate(
            config_path="configs/full_lora_config.yaml",
            weights_path="outputs/sam3_lora_full/best_lora_weights.pt",
            val_data_dir="/workspace/data2/valid"
        )

    Example (base SAM3 model):
        validate(
            config_path=None,
            weights_path=None,
            val_data_dir="/workspace/data2/valid",
            use_base_model=True
        )
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    plots_dir = Path(plots_dir) if plots_dir else Path("outputs") / "validation_plots"
    if plots:
        print(f"Plots will be written to: {plots_dir}")

    overlay_dir = (Path(overlay_dir) if overlay_dir
                   else plots_dir / "detections")
    if overlays:
        print(f"Detection overlays will be written to: {overlay_dir}")

    # Build model
    print("\nBuilding SAM3 model...")
    model = build_sam3_image_model(
        device=device.type,
        compile=False,
        load_from_HF=True,
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        eval_mode=False
    )

    # Load config for batch_size and other settings
    if use_base_model:
        # Use original SAM3 model without LoRA
        print("Using original SAM3 model (no LoRA)")
        stats = count_parameters(model)
        print(f"Total params: {stats['total_parameters']:,}")
        # Use default batch_size for base model
        batch_size = 1
    else:
        # Apply LoRA and load weights
        if config_path is None or weights_path is None:
            raise ValueError("config_path and weights_path are required when use_base_model=False")

        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Apply LoRA
        print("Applying LoRA configuration...")
        lora_cfg = config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        model = apply_lora_to_model(model, lora_config)

        # Load weights
        print(f"\nLoading LoRA weights from {weights_path}...")
        load_lora_weights(model, weights_path)

        stats = count_parameters(model)
        print(f"Trainable params: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")

        # Get batch_size from config
        batch_size = config["training"]["batch_size"]

    model.to(device)
    model.eval()

    # Load validation data directly from the specified directory
    print(f"\nLoading validation data from {val_data_dir}...")

    # Load COCO annotations directly
    ann_file = Path(val_data_dir) / "_annotations.coco.json"
    if not ann_file.exists():
        raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

    # Create a simple dataset class that loads from the directory directly:
    # val_data_dir IS the split directory, so no split subfolder is appended.
    class DirectCOCODataset(COCOSegmentDataset):
        def __init__(self, data_dir, all_class_queries=False):
            super().__init__(
                data_dir,
                split=None,
                all_class_queries=all_class_queries,
            )
            print(f"Loaded COCO dataset from {data_dir}")

    val_ds = DirectCOCODataset(val_data_dir, all_class_queries=classification)

    if num_samples:
        print(f"\n[INFO] Limiting validation to {num_samples} samples for debugging")

    def collate_fn(batch):
        return collate_fn_api(batch, dict_key="input", with_seg_masks=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,  # Enable parallel data loading
        pin_memory=True  # Faster GPU transfer
    )

    # Create matcher for loss computation
    matcher = BinaryHungarianMatcherV2(
        cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True
    )

    # Run validation
    print("\n" + "="*80)
    print("RUNNING VALIDATION")
    print("="*80)

    all_predictions = []
    all_image_ids = []      # one unique eval id per (image, prompt) pair
    all_eval_units = []     # parallel: {eval_id, src_idx, obj_ids} for GT building
    val_losses = []

    # Use automatic mixed precision for faster inference
    use_amp = device.type == 'cuda'

    with torch.no_grad():
        for batch_idx, batch_dict in enumerate(tqdm(val_loader, desc="Validation")):
            if num_samples and batch_idx * batch_size >= num_samples:
                break

            input_batch = batch_dict["input"]
            input_batch = move_to_device(input_batch, device)

            # Forward pass with optional AMP
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs_list = model(input_batch)
            else:
                outputs_list = model(input_batch)

            # Extract predictions
            with SAM3Output.iteration_mode(
                outputs_list, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
            ) as outputs_iter:
                final_stage = list(outputs_iter)[-1]
                final_outputs = final_stage[-1]

                # NOTE: the leading dim of pred_logits is the number of TEXT
                # PROMPTS (find-queries) in the batch, NOT the number of images.
                # An image with N categories contributes N prompts here. The old
                # code assumed one prompt per image (img_id = batch_idx*batch_size
                # + i), which both ran past len(dataset) and mis-assigned
                # predictions on multi-category data. Instead we read each
                # prompt's true source image and target objects from the batch
                # metadata/targets (same flattened order as the model output) and
                # treat every (image, prompt) pair as its own evaluation unit.
                num_prompts = final_outputs['pred_logits'].shape[0]

                # query_processing_order=0 -> stage 0 for the image model
                meta = input_batch.find_metadatas[0]
                tgt = input_batch.find_targets[0]
                src_img_ids = meta.original_image_id      # per-prompt source idx
                src_cat_ids = meta.original_category_id   # per-prompt class id
                num_boxes = tgt.num_boxes                 # per-prompt object count
                obj_ids_padded = tgt.object_ids_padded    # per-prompt object ids (-1 padded)

                for i in range(num_prompts):
                    eval_id = len(all_predictions)        # unique, sequential
                    src_idx = int(src_img_ids[i])
                    category_id = int(src_cat_ids[i])
                    n = int(num_boxes[i])
                    obj_ids = [int(o) for o in obj_ids_padded[i][:n].tolist()] if n > 0 else []

                    all_image_ids.append(eval_id)
                    all_eval_units.append({
                        'eval_id': eval_id,
                        'src_idx': src_idx,
                        'category_id': category_id,
                        'obj_ids': obj_ids,
                    })
                    pred_entry = {
                        'category_id': category_id,
                        'pred_logits': final_outputs['pred_logits'][i].detach().cpu(),
                        'pred_boxes': final_outputs['pred_boxes'][i].detach().cpu(),
                        'pred_masks': final_outputs['pred_masks'][i].detach().cpu()
                    }
                    # Prompt-level presence logit, needed to compute the joint
                    # detection score the way SAM3's official eval does.
                    if 'presence_logit_dec' in final_outputs:
                        pred_entry['presence_logit'] = (
                            final_outputs['presence_logit_dec'][i].detach().cpu()
                        )
                    all_predictions.append(pred_entry)

    print(f"\nCollected predictions for {len(all_predictions)} prompt units")

    # Compute metrics
    print("\n" + "="*80)
    print("COMPUTING METRICS")
    print("="*80)

    # Real category ids, so per-class (classification-aware) COCOeval works.
    categories_for_coco = [
        {'id': int(cid), 'name': str(name)}
        for cid, name in sorted(val_ds.categories.items())
        if int(cid) > 0
    ] or [{'id': 1, 'name': 'object'}]
    cat_id_to_name = {int(c['id']): c['name'] for c in categories_for_coco}

    # Prompt-level classification ("is this class present here?") over ALL
    # units, including the negative all-class prompts.
    if classification:
        print("\n" + "="*80)
        print("PROMPT-LEVEL CLASSIFICATION (presence head)")
        print("="*80)
        cls_metrics = compute_prompt_classification_metrics(
            all_eval_units, all_predictions, cat_id_to_name,
            threshold=classification_threshold
        )
        print(f"Decision threshold: {cls_metrics['threshold']:.2f} "
              f"on sigmoid(presence_logit)")
        print(f"Units scored: {cls_metrics['num_units']} "
              f"(skipped: {cls_metrics['num_skipped']})")
        print(f"\n{'class':<24}{'pos':>6}{'neg':>6}{'TP':>6}{'FP':>6}"
              f"{'FN':>6}{'TN':>6}{'acc':>8}{'P':>8}{'R':>8}{'F1':>8}")
        for row in cls_metrics['per_class'].values():
            print(f"{row['name'][:23]:<24}{row['pos']:>6}{row['neg']:>6}"
                  f"{row['tp']:>6}{row['fp']:>6}{row['fn']:>6}{row['tn']:>6}"
                  f"{row['accuracy']:>8.3f}{row['precision']:>8.3f}"
                  f"{row['recall']:>8.3f}{row['f1']:>8.3f}")
        print(f"\nMicro accuracy: {cls_metrics['accuracy']:.4f}")
        print(f"Macro precision/recall/F1: {cls_metrics['macro_precision']:.4f} / "
              f"{cls_metrics['macro_recall']:.4f} / {cls_metrics['macro_f1']:.4f}")

    # Units that own at least one GT object drive mAP/cgF1. Negative (all-class)
    # prompts have no GT and are covered by the classification metrics above.
    metric_units = [u for u in all_eval_units if u['obj_ids']]
    metric_predictions = [
        p for p, u in zip(all_predictions, all_eval_units) if u['obj_ids']
    ]
    metric_image_ids = [u['eval_id'] for u in metric_units]

    print(f"\n[INFO] {len(metric_units)} positive prompt units for mAP/cgF1 "
          f"(out of {len(all_eval_units)})")

    # Create COCO ground truth, one unit per (image, prompt) pair so multi-
    # category images are scored correctly (downsampled to 288×288 - fast!)
    print(f"\n[INFO] Creating per-prompt ground truth from validation dataset...")
    coco_gt_dict = create_coco_gt_per_prompt(
        val_ds,
        metric_units,
        mask_resolution=288,
        categories=categories_for_coco
    )

    # Check prediction scores (optional - can be commented out for speed)
    # print(f"\n[INFO] Analyzing prediction scores...")
    # all_scores = []
    # for p in all_predictions:
    #     if 'pred_logits' in p and len(p['pred_logits']) > 0:
    #         scores = torch.sigmoid(p['pred_logits']).squeeze(-1)
    #         all_scores.extend(scores.tolist())
    # if all_scores:
    #     print(f"[INFO] Prediction scores: min={min(all_scores):.4f}, max={max(all_scores):.4f}, mean={np.mean(all_scores):.4f}")

    # Convert predictions using SAM3's NMS pipeline or crack merging
    coco_predictions = convert_predictions_to_coco_format(
        metric_predictions,
        metric_image_ids,
        resolution=288,
        prob_threshold=prob_threshold,
        nms_iou_threshold=nms_iou,
        max_detections=100,
        merge_cracks=merge_cracks,
        merge_iou_threshold=merge_iou,
        merge_dilate=merge_dilate,
        use_presence=use_presence
    )

    if merge_cracks:
        print(f"\n[INFO] Total predictions after CRACK MERGING: {len(coco_predictions)}")
    else:
        print(f"\n[INFO] Total predictions after SAM3 NMS filtering: {len(coco_predictions)}")

    if len(coco_predictions) > 0:
        # Save temporary files for COCO evaluation
        import tempfile
        import os

        # Create temp directory for evaluation files
        temp_dir = tempfile.mkdtemp(prefix="sam3_eval_")
        gt_file = os.path.join(temp_dir, "gt.json")
        pred_file = os.path.join(temp_dir, "pred.json")

        with open(gt_file, 'w') as f:
            json.dump(coco_gt_dict, f)
        with open(pred_file, 'w') as f:
            json.dump(coco_predictions, f)

        # Compute mAP
        print("\n" + "="*80)
        print("COCO mAP EVALUATION")
        print("="*80)

        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_gt = COCO(str(gt_file))
                coco_dt = coco_gt.loadRes(str(pred_file))
                coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
                coco_eval.params.useCats = False
                coco_eval.evaluate()
                coco_eval.accumulate()

        # Print mAP results
        coco_eval.summarize()

        map_segm = coco_eval.stats[0]
        map50_segm = coco_eval.stats[1]
        map75_segm = coco_eval.stats[2]

        # ------------------------------------------------------------------
        # Classification-aware evaluation: one GT image per SOURCE image with
        # its REAL category ids, and every prompt's predictions labelled with
        # the category id its prompt asked for. COCOeval(useCats=True) charges
        # a correct-class hit as TP and a wrong-class hit as FP + FN, so
        # misclassification is penalised instead of hidden.
        # ------------------------------------------------------------------
        per_class_ap = {}
        cls_map_segm = cls_map50_segm = 0.0
        cm_matrix, cm_names = None, None
        coco_eval_cat = None
        if classification:
            print("\n" + "="*80)
            print("CLASSIFICATION-AWARE mAP (real category ids, useCats=True)")
            print("="*80)

            src_indices = sorted({int(u['src_idx']) for u in all_eval_units})
            gt_cls = create_coco_gt_per_image(
                val_ds, src_indices,
                mask_resolution=288,
                categories=categories_for_coco
            )

            cls_units = [u for u in all_eval_units if int(u['category_id']) > 0]
            cls_predictions = [
                p for p, u in zip(all_predictions, all_eval_units)
                if int(u['category_id']) > 0
            ]
            cls_image_ids = [int(u['src_idx']) for u in cls_units]
            coco_predictions_cls = convert_predictions_to_coco_format(
                cls_predictions, cls_image_ids,
                resolution=288,
                prob_threshold=prob_threshold,
                nms_iou_threshold=nms_iou,
                max_detections=100,
                merge_cracks=merge_cracks,
                merge_iou_threshold=merge_iou,
                merge_dilate=merge_dilate,
                use_presence=use_presence
            )

            if len(coco_predictions_cls) > 0:
                cls_gt_file = os.path.join(temp_dir, "gt_cls.json")
                cls_pred_file = os.path.join(temp_dir, "pred_cls.json")
                with open(cls_gt_file, 'w') as f:
                    json.dump(gt_cls, f)
                with open(cls_pred_file, 'w') as f:
                    json.dump(coco_predictions_cls, f)

                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        coco_gt_c = COCO(str(cls_gt_file))
                        coco_dt_c = coco_gt_c.loadRes(str(cls_pred_file))
                        coco_eval_cat = COCOeval(coco_gt_c, coco_dt_c, 'segm')
                        # useCats stays True -> one AP per real category
                        coco_eval_cat.evaluate()
                        coco_eval_cat.accumulate()

                coco_eval_cat.summarize()
                per_class_ap = print_per_class_ap(coco_eval_cat, cat_id_to_name)
                cls_map_segm = float(coco_eval_cat.stats[0])
                cls_map50_segm = float(coco_eval_cat.stats[1])

                if plots:
                    cm_matrix, cm_names = build_confusion_matrix(
                        coco_predictions_cls, gt_cls, cat_id_to_name,
                        iou_threshold=cm_iou, resolution=288,
                        use_boxes=cm_boxes,
                    )
                    print("\nConfusion matrix (rows = predicted, cols = true; "
                          f"IoU >= {cm_iou:.2f} on "
                          f"{'boxes' if cm_boxes else 'masks'})")
                    total = cm_matrix.sum(axis=0)
                    norm = cm_matrix / (total.reshape(1, -1) + 1e-9)
                    print(f"{'label':<24}{'accuracy':>12}")
                    for i, name in enumerate(cm_names):
                        acc = 0.0 if np.isnan(norm[i, i]) else float(norm[i, i])
                        print(f"{name[:23]:<24}{acc:>12.4f}")

                if overlays:
                    target_classes = overlay_classes or [
                        c['name'] for c in categories_for_coco
                    ]
                    print("\n" + "="*80)
                    print("DETECTION OVERLAYS (green=TP, red=FN, yellow=FP)")
                    print("="*80)
                    save_class_overlays(
                        val_ds, coco_predictions_cls, gt_cls,
                        target_classes, overlay_dir,
                        iou_threshold=overlay_iou, resolution=288,
                    )
            else:
                print("[WARN] No classification-aware predictions to evaluate")

        # Compute cgF1
        print("\n" + "="*80)
        print("cgF1 EVALUATION")
        print("="*80)

        cgf1_evaluator = CGF1Evaluator(
            gt_path=str(gt_file),
            iou_type='segm',
            verbose=True
        )
        cgf1_results = cgf1_evaluator.evaluate(str(pred_file))

        cgf1 = cgf1_results.get('cgF1_eval_segm_cgF1', 0.0)
        cgf1_50 = cgf1_results.get('cgF1_eval_segm_cgF1@0.5', 0.0)
        cgf1_75 = cgf1_results.get('cgF1_eval_segm_cgF1@0.75', 0.0)

        # Compute pixel-level metrics (fragmentation-insensitive)
        pixel_results = None
        if pixel_metrics:
            print("\n" + "="*80)
            print("PIXEL-LEVEL (SEMANTIC) EVALUATION")
            print("="*80)
            pixel_results = compute_pixel_metrics(coco_gt_dict, coco_predictions)

        # Print summary
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"mAP (IoU 0.50:0.95, class-agnostic): {map_segm:.4f}")
        print(f"mAP@50: {map50_segm:.4f}")
        print(f"mAP@75: {map75_segm:.4f}")
        print(f"cgF1 (IoU 0.50:0.95): {cgf1:.4f}")
        print(f"cgF1@50: {cgf1_50:.4f}")
        print(f"cgF1@75: {cgf1_75:.4f}")
        if classification:
            print("-"*80)
            print("Classification-aware (real category ids, scored per class):")
            print(f"mAP (IoU 0.50:0.95): {cls_map_segm:.4f}")
            print(f"mAP@50: {cls_map50_segm:.4f}")
            print(f"Prompt classification accuracy: {cls_metrics['accuracy']:.4f}")
            print(f"Prompt macro P/R/F1: {cls_metrics['macro_precision']:.4f} / "
                  f"{cls_metrics['macro_recall']:.4f} / {cls_metrics['macro_f1']:.4f}")
        if pixel_results is not None:
            print("-"*80)
            print("Pixel-level metrics (union of masks per prompt; robust to")
            print("a single crack being predicted as multiple fragments):")
            print(f"Pixel IoU:       {pixel_results['pixel_iou']:.4f}")
            print(f"Pixel Precision: {pixel_results['pixel_precision']:.4f}")
            print(f"Pixel Recall:    {pixel_results['pixel_recall']:.4f}")
            print(f"Mean per-prompt IoU: {pixel_results['mean_unit_iou']:.4f}")
        print("="*80)

        # Plots: confusion matrix, per-class bars, PR curve, metric summary
        if plots:
            summary = {
                'mAP': float(map_segm),
                'mAP50': float(map50_segm),
                'cgF1': float(cgf1),
            }
            if classification:
                summary['cls_mAP'] = float(cls_map_segm)
                summary['cls_mAP50'] = float(cls_map50_segm)
                summary['prompt_acc'] = float(cls_metrics['accuracy'])
            if pixel_results is not None:
                summary['pixel_IoU'] = float(pixel_results['pixel_iou'])

            written = save_validation_plots(
                plots_dir,
                confusion_matrix=cm_matrix,
                confusion_names=cm_names,
                per_class_ap=per_class_ap,
                cls_metrics=cls_metrics if classification else None,
                coco_eval=coco_eval_cat,
                cat_id_to_name=cat_id_to_name,
                summary=summary,
            )
            print(f"\n[INFO] Plots saved to {plots_dir} ({len(written)} files)")
            for plot_path in written:
                print(f"  - {plot_path}")

        # Cleanup temporary files
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

    else:
        print("\n[ERROR] No predictions generated! Cannot compute metrics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone validation script for SAM3 LoRA model with full metrics (mAP, cgF1) and SAM3 NMS"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (for LoRA settings). Not required if --use-base-model is set."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to LoRA weights file. Not required if --use-base-model is set."
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        required=True,
        help="Direct path to validation data directory containing _annotations.coco.json (e.g., /workspace/data2/valid)"
    )
    parser.add_argument(
        "--use-base-model",
        action="store_true",
        help="Use original SAM3 model without LoRA (for baseline comparison)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Limit validation to N samples (for debugging)"
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.1,
        help="Probability threshold for filtering predictions (default: 0.3)"
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold (default: 0.7)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Enable aggressive merging of overlapping segments (recommended for crack detection)"
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.15,
        help="IoU threshold for merging overlapping predictions (default: 0.15, lower = more aggressive)"
    )
    parser.add_argument(
        "--merge-dilate",
        type=int,
        default=0,
        help="Dilation radius in pixels (at 288 mask resolution) for --merge: "
             "disjoint fragments within ~2x this distance are joined into ONE "
             "instance before matching. Fixes a single GT crack predicted as "
             "several portions. 0 disables (default). Try 3-5 for cracks. "
             "Requires --merge."
    )
    parser.add_argument(
        "--pixel-metrics",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="Also report pixel-level (semantic) IoU/precision/recall computed "
             "on the union of masks per prompt — insensitive to a crack being "
             "split into multiple predicted fragments: True or False (default: True)"
    )
    parser.add_argument(
        "--no-presence",
        action="store_true",
        help="Score with raw pred_logits instead of multiplying in the presence "
             "score. SAM3's official eval uses presence; disabling it makes the "
             "zero-shot baseline collapse to ~0 mAP."
    )
    parser.add_argument(
        "--no-classification",
        action="store_true",
        help="Disable classification-aware evaluation. By default every class of "
             "the split is queried on every image, predictions/GT keep their real "
             "category ids, and the script reports per-class AP plus prompt-level "
             "presence classification (accuracy / P / R / F1)."
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.5,
        help="Presence probability above which a class is declared present in the "
             "prompt-level classification metrics (default: 0.5)"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip saving the confusion matrix and metric plots"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Directory for the saved plots (default: outputs/validation_plots)"
    )
    parser.add_argument(
        "--cm-iou",
        type=float,
        default=0.25,
        help="Mask/box IoU threshold for matching predictions to GT when building "
             "the confusion matrix (default: 0.25)"
    )
    parser.add_argument(
        "--cm-boxes",
        action="store_true",
        default=True,
        help="Build the confusion matrix from BOX IoU (default). "
             "Box IoU is far more forgiving for thin objects such as needle."
    )
    parser.add_argument(
        "--cm-masks",
        action="store_false",
        dest="cm_boxes",
        help="Match the confusion matrix on MASK IoU instead of boxes"
    )
    parser.add_argument(
        "--overlay-classes",
        nargs="+",
        default=None,
        help="Classes to render in the GT/prediction overlays "
             "(default: all classes in the split), e.g. --overlay-classes clamp forceps"
    )
    parser.add_argument(
        "--overlay-dir",
        type=str,
        default=None,
        help="Root directory for overlay images (default: <plots-dir>/detections)"
    )
    parser.add_argument(
        "--overlay-iou",
        type=float,
        default=0.1,
        help="Box IoU threshold for TP/FN/FP assignment in overlays (default: 0.5)"
    )
    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip saving the GT/prediction overlay images"
    )
    args = parser.parse_args()

    # Validate argument combinations
    if not args.use_base_model:
        if args.config is None or args.weights is None:
            parser.error("--config and --weights are required when not using --use-base-model")
    if args.merge_dilate > 0 and not args.merge:
        parser.error("--merge-dilate requires --merge (dilation only applies in merging mode)")

    validate(
        config_path=args.config,
        weights_path=args.weights,
        val_data_dir=args.val_data_dir,
        num_samples=args.num_samples,
        prob_threshold=args.prob_threshold,
        nms_iou=args.nms_iou,
        merge_cracks=args.merge,
        merge_iou=args.merge_iou,
        merge_dilate=args.merge_dilate,
        use_base_model=args.use_base_model,
        use_presence=not args.no_presence,
        pixel_metrics=args.pixel_metrics,
        classification=not args.no_classification,
        classification_threshold=args.classification_threshold,
        plots=not args.no_plots,
        plots_dir=args.plots_dir,
        cm_iou=args.cm_iou,
        cm_boxes=args.cm_boxes,
        overlays=not args.no_overlays,
        overlay_classes=args.overlay_classes,
        overlay_dir=args.overlay_dir,
        overlay_iou=args.overlay_iou
    )

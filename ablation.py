"""
Ablation Study Runner for Cell Instance Segmentation
=====================================================
每個 component 可以透過 flag 開關，跑完自動記錄結果到 CSV。

Usage:
  # 單次實驗
  python ablation.py --exp_name baseline \
  --no_a2fpn --no_pointrend --no_strong_aug --no_tta

  # 跑全部消融組合（用 run_ablation.sh）
  bash run_ablation.sh
"""

import matplotlib.pyplot as plt
import os
import json
import random
import tempfile
import argparse
import csv
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import tifffile
import skimage.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import ResNeXt101_32X8D_Weights
from torchvision.ops import nms
from torch.utils.data import Dataset, DataLoader, Subset

from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import matplotlib
matplotlib.use('Agg')

# ─────────────────────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────────────────────


def get_args():
    p = argparse.ArgumentParser(
        description='Ablation Study for Cell Instance Segmentation')

    # ── Experiment identity ──────────────────────────────────────────
    p.add_argument('--exp_name', type=str, required=True,
                   help='Experiment name (used for saving results)')
    p.add_argument('--output_dir', type=str, default='./ablation_output')
    p.add_argument('--result_csv', type=str, default='./ablation_results.csv',
                   help='CSV file to append results to')

    # ── Paths ────────────────────────────────────────────────────────
    p.add_argument('--train_dir', type=str, default='./train')
    p.add_argument('--gpu_ids', type=int, nargs='+', default=None)

    # ── Training ─────────────────────────────────────────────────────
    p.add_argument('--epochs', type=int, default=15,
                   help='Shorter epochs for ablation (default: 15)')
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--val_split', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)

    # ════════════════════════════════════════════════════════════════
    # Ablation flags — each flag DISABLES one component
    # ════════════════════════════════════════════════════════════════

    # Backbone
    p.add_argument('--no_resnext', action='store_true',
                   help='Use ResNet-50 instead of ResNeXt-101 backbone')

    # Neck
    p.add_argument('--no_a2fpn', action='store_true',
                   help='Disable A²-FPN attention (use plain FPN)')

    # Mask head
    p.add_argument('--no_pointrend', action='store_true',
                   help='Use standard mask head instead of PointRend')

    # Anchors
    p.add_argument(
        '--no_small_anchors',
        action='store_true',
        help=(
            'Use default anchor sizes instead of small cell-optimized anchors'
        ))

    # Augmentation
    p.add_argument('--no_strong_aug', action='store_true',
                   help='Use only H/V flip (no rotation, no color jitter)')

    p.add_argument('--no_color_jitter', action='store_true',
                   help='Disable color jitter only (keep rotation)')

    p.add_argument('--no_rotation', action='store_true',
                   help='Disable 90° rotation only (keep color jitter)')

    # Inference
    p.add_argument('--no_tta', action='store_true',
                   help='Disable Test Time Augmentation at validation')

    # ── Predict mode ────────────────────────────────────────────
    p.add_argument('--mode', type=str, default='train',
                   choices=['train', 'predict'],
                   help='train or predict')
    p.add_argument('--test_dir', type=str, default='./test')
    p.add_argument('--id_json', type=str,
                   default='./test_image_name_to_ids.json')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='checkpoint path for prediction')
    p.add_argument('--score_thresh', type=float, default=0.5)

    return p.parse_args()


CLASS_NAMES = ['background', 'class1', 'class2', 'class3', 'class4']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def encode_mask(binary_mask):
    arr = np.asfortranarray(binary_mask).astype(np.uint8)
    rle = mask_utils.encode(arr)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def read_maskfile(filepath):
    return sio.imread(str(filepath))


def mask_to_bbox(binary_mask):
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [float(cmin), float(rmin), float(cmax), float(rmax)]


def collate_fn(batch):
    return tuple(zip(*batch))


def _to_rgb(img_np):
    img_np = np.squeeze(img_np)
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.ndim == 3:
        if img_np.shape[2] > 4 and img_np.shape[0] <= 4:
            img_np = img_np.transpose(1, 2, 0)
        c = img_np.shape[2]
        if c == 1:
            img_np = np.concatenate([img_np] * 3, axis=-1)
        elif c == 2:
            img_np = np.stack([img_np[:, :, 0]] * 3, axis=-1)
        elif c == 3:
            pass
        else:
            img_np = img_np[:, :, :3]
    if img_np.dtype != np.uint8:
        mn, mx = float(img_np.min()), float(img_np.max())
        img_np = ((img_np.astype(np.float32) - mn) / (mx - mn + 1e-8)
                  * 255).clip(0, 255).astype(np.uint8)
    assert img_np.shape[2] == 3
    return img_np


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation (controllable)
# ─────────────────────────────────────────────────────────────────────────────

class Augmentation:
    def __init__(self, p_flip=0.5, p_rotate=0.5, p_color=0.4,
                 use_rotation=True, use_color_jitter=True):
        self.p_flip = p_flip
        self.p_rotate = p_rotate if use_rotation else 0.0
        self.p_color = p_color if use_color_jitter else 0.0

    def __call__(self, image, target):
        # H-flip
        if random.random() < self.p_flip:
            _, _, W = image.shape
            image = image.flip(-1)
            b = target['boxes'].clone()
            b[:, [0, 2]] = W - b[:, [2, 0]]
            target['boxes'] = b
            target['masks'] = target['masks'].flip(-1)
        # V-flip
        if random.random() < self.p_flip:
            _, H, _ = image.shape
            image = image.flip(-2)
            b = target['boxes'].clone()
            b[:, [1, 3]] = H - b[:, [3, 1]]
            target['boxes'] = b
            target['masks'] = target['masks'].flip(-2)
        # 90° rotation
        if random.random() < self.p_rotate:
            k = random.choice([1, 2, 3])
            image = torch.rot90(image, k, dims=[-2, -1])
            target['masks'] = torch.rot90(target['masks'], k, dims=[-2, -1])
            new_boxes = []
            for m in target['masks']:
                b = mask_to_bbox(m.numpy())
                new_boxes.append(b if b else [0., 0., 1., 1.])
            target['boxes'] = torch.tensor(new_boxes, dtype=torch.float32)
        # Color jitter
        if random.random() < self.p_color:
            image = TF.adjust_brightness(image, 1 + random.uniform(-0.3, 0.3))
            image = TF.adjust_contrast(image, 1 + random.uniform(-0.3, 0.3))
            image = TF.adjust_saturation(image, 1 + random.uniform(-0.3, 0.3))
            image = image.clamp(0., 1.)
        return image, target


class NoAugmentation:
    def __call__(self, image, target): return image, target


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CellDataset(Dataset):
    CLASS_MAP = {'class1': 1, 'class2': 2, 'class3': 3, 'class4': 4}

    def __init__(self, root_dir, augmentation=None):
        self.root = Path(root_dir)
        self.aug = augmentation or NoAugmentation()
        self.samples = sorted([d for d in self.root.iterdir()
                               if d.is_dir() and (d / 'image.tif').exists()])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sd = self.samples[idx]
        img_np = _to_rgb(tifffile.imread(str(sd / 'image.tif')))
        img_tensor = torch.from_numpy(
            img_np.transpose(
                2, 0, 1)).float() / 255.0

        boxes, labels, masks = [], [], []
        for cls_name, cls_id in self.CLASS_MAP.items():
            mp = sd / f'{cls_name}.tif'
            if not mp.exists():
                continue
            ma = read_maskfile(mp)
            for inst_id in np.unique(ma):
                if inst_id == 0:
                    continue
                binary = (ma == inst_id).astype(np.uint8)
                bbox = mask_to_bbox(binary)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = bbox
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                boxes.append(bbox)
                labels.append(cls_id)
                masks.append(binary)

        H, W = img_tensor.shape[1], img_tensor.shape[2]
        if len(boxes) == 0:
            target = {'boxes': torch.zeros((0, 4), dtype=torch.float32),
                      'labels': torch.zeros((0,), dtype=torch.int64),
                      'masks': torch.zeros((0, H, W), dtype=torch.uint8)}
        else:
            target = {
                'boxes': torch.tensor(
                    boxes, dtype=torch.float32), 'labels': torch.tensor(
                    labels, dtype=torch.int64), 'masks': torch.tensor(
                    np.array(masks), dtype=torch.uint8)}

        img_tensor, target = self.aug(img_tensor, target)
        return img_tensor, target


# ─────────────────────────────────────────────────────────────────────────────
# Model Components (all toggleable)
# ─────────────────────────────────────────────────────────────────────────────

# ── A²-FPN ──────────────────────────────────────────────────────────────────
class ChannelAttention(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        mid = max(c // r, 16)
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                nn.Linear(c, mid), nn.ReLU(True),
                                nn.Linear(mid, c), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)


class A2FPN(nn.Module):
    """A²-FPN: Attention Aggregation FPN. Hu et al. CVPR 2021."""

    def __init__(self, backbone_with_fpn):
        super().__init__()
        self.backbone = backbone_with_fpn
        c = backbone_with_fpn.out_channels
        self.out_channels = c
        self.attention = nn.ModuleDict({k: ChannelAttention(c)
                                        for k in ['0', '1', '2', '3', 'pool']})

    def forward(self, x):
        feats = self.backbone(x)
        result = {}
        for k, f in feats.items():
            attn = (
                self.attention[k]
                if k in self.attention
                else self.attention['pool']
            )
            result[k] = attn(f)
        return result


# ── PointRend Mask Head ──────────────────────────────────────────────────────
class PointRendMaskHead(nn.Module):
    """PointRend mask head. Kirillov et al. CVPR 2020."""

    def __init__(
            self,
            in_channels=256,
            num_classes=5,
            num_points=196,
            hidden=256):
        super().__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        self.in_proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden,
                1),
            nn.ReLU(True))
        layers = []
        for _ in range(4):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(hidden,
                                      hidden,
                                      2,
                                      stride=2),
                   nn.ReLU(True),
                   nn.Conv2d(hidden,
                             num_classes,
                             1)]
        self.coarse_head = nn.Sequential(*layers)
        self.point_mlp = nn.Sequential(
            nn.Linear(num_classes + hidden, 256), nn.ReLU(True),
            nn.Linear(256, 256), nn.ReLU(True),
            nn.Linear(256, num_classes))

    def _uncertain(self, logits, n):
        N, C, H, W = logits.shape
        unc = -logits.abs().max(dim=1).values.view(N, -1)
        n = min(n, H * W)
        _, idx = unc.topk(n, dim=1)
        y = (idx // W).float() / max(H - 1, 1) * 2 - 1
        x = (idx % W).float() / max(W - 1, 1) * 2 - 1
        return torch.stack([x, y], dim=-1)

    def forward(self, features):
        x = self.in_proj(features)
        coarse = self.coarse_head(x)
        if coarse.shape[0] == 0:
            return coarse
        pts = self._uncertain(coarse, self.num_points)
        grid = pts.unsqueeze(2)
        cp = F.grid_sample(
            coarse,
            grid,
            align_corners=True,
            mode='bilinear').squeeze(3).permute(
            0,
            2,
            1)
        fp = F.grid_sample(
            x,
            grid,
            align_corners=True,
            mode='bilinear').squeeze(3).permute(
            0,
            2,
            1)
        refined = self.point_mlp(torch.cat([cp, fp], dim=-1))
        out = coarse.clone()
        N, C, H, W = coarse.shape
        px = ((pts[..., 0] + 1) / 2 * (W - 1)).long().clamp(0, W - 1)
        py = ((pts[..., 1] + 1) / 2 * (H - 1)).long().clamp(0, H - 1)
        for n in range(N):
            out[n, :, py[n], px[n]] = refined[n].T
        return out


# ── MaskRCNN Wrapper (for training) ─────────────────────────────────────────
class MaskRCNNWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            loss_dict = self.model(images, targets)
            return sum(loss_dict.values()), {
                k: v.detach() for k, v in loss_dict.items()}
        return self.model(images)


# ─────────────────────────────────────────────────────────────────────────────
# Model Builder (fully controlled by args)
# ─────────────────────────────────────────────────────────────────────────────

def build_model(args) -> nn.Module:
    num_classes = 5

    # ── 1. Backbone ──────────────────────────────────────────────────
    if args.no_resnext:
        # Baseline: ResNet-50
        backbone_fpn = resnet_fpn_backbone(
            backbone_name='resnet50',
            weights='DEFAULT',
            trainable_layers=3)
    else:
        # Improved: ResNeXt-101  (Xie et al. CVPR 2017)
        backbone_fpn = resnet_fpn_backbone(
            backbone_name='resnext101_32x8d',
            weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1,
            trainable_layers=3)

    # ── 2. Neck (FPN) ────────────────────────────────────────────────
    if args.no_a2fpn:
        backbone = backbone_fpn          # plain FPN
    else:
        backbone = A2FPN(backbone_fpn)   # A²-FPN  (Hu et al. CVPR 2021)

    # ── 3. Anchors ───────────────────────────────────────────────────
    if args.no_small_anchors:
        anchor_gen = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,), (512,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    else:
        # Small-cell optimized anchors
        anchor_gen = AnchorGenerator(
            sizes=((4, 8), (8, 16), (16, 32), (32, 64), (64, 128)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5)

    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        box_detections_per_img=150,
        min_size=400, max_size=600)

    # ── 4. Mask Head ─────────────────────────────────────────────────
    if not args.no_pointrend:
        # PointRend  (Kirillov et al. CVPR 2020)
        model.roi_heads.mask_predictor = PointRendMaskHead(
            in_channels=256, num_classes=num_classes,
            num_points=196, hidden=256)
    # else: keep default MaskRCNN mask head

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def build_coco_gt(val_loader):
    coco_gt = {'images': [], 'annotations': [], 'categories': [
        {'id': i, 'name': CLASS_NAMES[i]} for i in range(1, 5)]}
    ann_id = 1
    image_ids = []
    for img_id, (images, targets) in enumerate(val_loader, 1):
        img = images[0]
        tgt = targets[0]
        H, W = img.shape[1], img.shape[2]
        coco_gt['images'].append({'id': img_id, 'height': H, 'width': W})
        image_ids.append(img_id)
        for box, label, m in zip(tgt['boxes'].numpy(),
                                 tgt['labels'].numpy(),
                                 tgt['masks'].numpy()):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            coco_gt['annotations'].append({
                'id': ann_id, 'image_id': img_id, 'category_id': int(label),
                'bbox': [float(x1), float(y1), float(w), float(h)],
                'segmentation': encode_mask(m.astype(np.uint8)),
                'area': float(w * h), 'iscrowd': 0})
            ann_id += 1
    return coco_gt, image_ids


@torch.no_grad()
def predict_single(
        model,
        img_tensor,
        device,
        score_thresh=0.05,
        use_tta=False):
    model.eval()
    if not use_tta:
        preds = model([img_tensor.to(device)])[0]
        return preds['scores'].cpu(), preds['labels'].cpu(), \
            preds['masks'].cpu().squeeze(1), preds['boxes'].cpu()

    all_s, all_l, all_m, all_b = [], [], [], []
    for hf, vf in [(False, False), (True, False), (False, True), (True, True)]:
        img = img_tensor.clone()
        if hf:
            img = img.flip(-1)
        if vf:
            img = img.flip(-2)
        pred = model([img.to(device)])[0]
        s = pred['scores'].cpu()
        lb = pred['labels'].cpu()
        m = pred['masks'].cpu().squeeze(1)
        b = pred['boxes'].cpu()
        if hf:
            m = m.flip(-1)
            W = img.shape[-1]
            b[:, [0, 2]] = W - b[:, [2, 0]]
        if vf:
            m = m.flip(-2)
            H = img.shape[-2]
            b[:, [1, 3]] = H - b[:, [3, 1]]
        keep = s >= score_thresh
        all_s.append(s[keep])
        all_l.append(lb[keep])
        all_m.append(m[keep])
        all_b.append(b[keep])
    if not any(len(x) > 0 for x in all_s):
        return torch.tensor([]), torch.tensor([]), \
            torch.zeros((0,) + img_tensor.shape[1:]), torch.tensor([])
    all_s = torch.cat(all_s)
    all_l = torch.cat(all_l)
    all_m = torch.cat(all_m)
    all_b = torch.cat(all_b)
    keep = nms(all_b, all_s, 0.5)
    return all_s[keep], all_l[keep], all_m[keep], all_b[keep]


@torch.no_grad()
def evaluate(model, val_loader, device, use_tta=False):
    model.eval()
    coco_gt_dict, image_ids = build_coco_gt(val_loader)
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(coco_gt_dict, f)
        gt_path = f.name
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(gt_path)

    results = []
    for img_id, (images, _) in zip(image_ids, val_loader):
        scores, labels, masks, _ = predict_single(
            model, images[0], device, score_thresh=0.05, use_tta=use_tta)
        keep = scores >= 0.05
        for sc, lb, m in zip(scores[keep], labels[keep], masks[keep]):
            results.append({'image_id': img_id, 'category_id': int(lb),
                            'segmentation': encode_mask(
                                (m.numpy() > 0.5).astype(np.uint8)),
                            'score': float(sc)})
    if not results:
        return 0.0, 0.0

    coco_dt = coco_gt.loadRes(results)

    import io as _io
    import contextlib as _ctx

    # ── AP50 only (reliable) ─────────────────────────────────────────
    ev50 = COCOeval(coco_gt, coco_dt, 'segm')
    ev50.params.iouThrs = np.array([0.50])
    ev50.params.maxDets = [1, 10, 100]
    ev50.evaluate()
    ev50.accumulate()
    buf = _io.StringIO()
    with _ctx.redirect_stdout(buf):
        ev50.summarize()
    print(buf.getvalue().strip())
    ap50 = float(ev50.stats[0]) if len(ev50.stats) > 0 else 0.0

    # ── AP50:95 ──────────────────────────────────────────────────────
    try:
        ev = COCOeval(coco_gt, coco_dt, 'segm')
        ev.evaluate()
        ev.accumulate()
        ap5095 = float(ev.stats[0]) if len(ev.stats) > 0 else 0.0
    except Exception:
        ap5095 = 0.0

    return ap50, ap5095


# ─────────────────────────────────────────────────────────────────────────────
# CSV Logger
# ─────────────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    'timestamp', 'exp_name',
    # ablation flags
    'backbone', 'a2fpn', 'pointrend', 'small_anchors',
    'strong_aug', 'color_jitter', 'rotation', 'tta',
    # results
    'best_ap50', 'best_ap5095', 'best_epoch',
    'train_time_min', 'epochs', 'final_loss',
]


def log_result(csv_path, row: dict):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    exists = Path(csv_path).exists()
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f'\n  ✓ Result logged to {csv_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    if args.gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            str(g) for g in args.gpu_ids)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp_dir = Path(args.output_dir) / args.exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── Print experiment config ──────────────────────────────────────
    print(f'\n{"=" * 65}')
    print(f'  Experiment : {args.exp_name}')
    print(f'  Device     : {device}')
    print(f'  {"─" * 55}')
    backbone_str = 'ResNet-50 (baseline)' if args.no_resnext else 'ResNeXt-101'
    print(f'  Backbone   : {backbone_str}')
    print(f'  A²-FPN     : {"✗ disabled" if args.no_a2fpn else "✓ enabled"}')
    val_pr = '\u2713 enabled'
    val_dis = '\u2717 disabled'
    print(f'  PointRend  : {val_dis if args.no_pointrend else val_pr}')
    print(f'  Sm.Anchors : {val_dis if args.no_small_anchors else val_pr}')
    no_cj = args.no_color_jitter or args.no_strong_aug
    print(f'  Color Jit  : {val_dis if no_cj else val_pr}')
    no_rot = args.no_rotation or args.no_strong_aug
    print(f'  Rotation   : {val_dis if no_rot else val_pr}')
    print(f'  TTA        : {val_dis if args.no_tta else val_pr}')
    print(f'{"=" * 65}\n')

    set_seed(args.seed)

    # ── Augmentation ─────────────────────────────────────────────────
    use_rotation = not (args.no_rotation or args.no_strong_aug)
    use_color = not (args.no_color_jitter or args.no_strong_aug)
    train_aug = Augmentation(
        use_rotation=use_rotation,
        use_color_jitter=use_color)

    # ── Datasets ─────────────────────────────────────────────────────
    full_ds = CellDataset(args.train_dir)
    n = len(full_ds)
    n_val = max(1, int(n * args.val_split))
    idx = list(range(n))
    random.shuffle(idx)
    train_idx, val_idx = idx[n_val:], idx[:n_val]

    train_set = Subset(CellDataset(args.train_dir, train_aug), train_idx)
    val_set = Subset(CellDataset(args.train_dir, NoAugmentation()), val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2)
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2)

    n_tr, n_val = len(train_set), len(val_set)
    print(f'Dataset: {n} images -> train={n_tr}, val={n_val}\n')

    # ── Model ─────────────────────────────────────────────────────────
    raw_model = build_model(args)
    raw_model.to(device)
    model = MaskRCNNWrapper(raw_model)

    n_params = sum(p.numel()
                   for p in raw_model.parameters() if p.requires_grad)
    print(f'Params: {n_params / 1e6:.1f} M\n')

    optimizer = torch.optim.AdamW(
        [p for p in raw_model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.6), int(args.epochs * 0.85)],
        gamma=0.1)

    # ── Training loop ─────────────────────────────────────────────────
    best_ap50 = 0.0
    best_ap5095 = 0.0
    best_epoch = 0
    history_loss = []
    history_ap50 = []
    history_ap5095 = []
    final_loss = 0.0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        print(
            f'\n\u2500\u2500 Epoch {epoch}/{args.epochs}'
            f'  (lr={optimizer.param_groups[0]["lr"]:.1e}) \u2500\u2500'),

        for step, (images, targets) in enumerate(train_loader, 1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()}
                       for t in targets]

            result = model(images, targets)
            if isinstance(result, (tuple, list)):
                losses, loss_dict = result[0], result[1]
                if losses.dim() > 0:
                    losses = losses.mean()
            else:
                loss_dict = result
                losses = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 5.0)
            optimizer.step()
            torch.cuda.empty_cache()
            epoch_loss += losses.item()

            if step % 20 == 0 or step == len(train_loader):
                detail = '  '.join(
                    f'{k}={v.item():.3f}'
                    for k, v in loss_dict.items())
                print(
                    f'  [Step {step:3d}/{len(train_loader)}]'
                    f'  total={losses.item():.4f}  ({detail})')

        avg_loss = epoch_loss / len(train_loader)
        history_loss.append(avg_loss)
        final_loss = avg_loss
        scheduler.step()
        print(f'\n  ► Epoch {epoch} avg_loss={avg_loss:.4f}')

        # ── Validation ────────────────────────────────────────────────
        print('\n  ── Validation ──')
        use_tta = not args.no_tta
        ap50, ap5095 = evaluate(raw_model, val_loader, device, use_tta=use_tta)
        history_ap50.append(ap50)
        history_ap5095.append(ap5095)
        print(f'  ► AP50={ap50:.4f}  AP50:95={ap5095:.4f}')

        if ap50 > best_ap50:
            best_ap50 = ap50
            best_ap5095 = ap5095
            best_epoch = epoch
            torch.save(raw_model.state_dict(), exp_dir / 'best_model.pth')
            print(f'  ✓ Saved best model (AP50={best_ap50:.4f})')

    train_time = (time.time() - t_start) / 60

    # ── Plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(range(1, len(history_loss) + 1), history_loss, marker='o')
    axes[0].set(title='Training Loss', xlabel='Epoch', ylabel='Loss')

    axes[1].plot(
        range(
            1,
            len(history_ap50) +
            1),
        history_ap50,
        marker='o',
        color='orange')
    axes[1].set(title='Val AP50', xlabel='Epoch', ylabel='AP50')

    axes[2].plot(
        range(
            1,
            len(history_ap5095) +
            1),
        history_ap5095,
        marker='o',
        color='green')
    axes[2].set(title='Val AP50:95', xlabel='Epoch', ylabel='AP')

    plt.suptitle(args.exp_name, fontsize=12)
    plt.tight_layout()
    plt.savefig(exp_dir / 'curves.png', dpi=150)
    print(f'\n  Curves saved to {exp_dir}/curves.png')

    # ── Summary ───────────────────────────────────────────────────────
    print(f'\n{"=" * 65}')
    print(f'  Experiment  : {args.exp_name}')
    print(f'  Best AP50   : {best_ap50:.4f}  (epoch {best_epoch})')
    print(f'  Best AP5095 : {best_ap5095:.4f}')
    print(f'  Train time  : {train_time:.1f} min')
    print(f'{"=" * 65}\n')

    # ── Log to CSV ────────────────────────────────────────────────────
    log_result(args.result_csv, {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'exp_name': args.exp_name,
        'backbone': 'ResNet50' if args.no_resnext else 'ResNeXt101',
        'a2fpn': 'off' if args.no_a2fpn else 'on',
        'pointrend': 'off' if args.no_pointrend else 'on',
        'small_anchors': 'off' if args.no_small_anchors else 'on',
        'strong_aug': 'off' if args.no_strong_aug else 'on',
        'color_jitter': (
            'off' if (args.no_color_jitter or args.no_strong_aug)
            else 'on'),
        'rotation': (
            'off' if (args.no_rotation or args.no_strong_aug)
            else 'on'),
        'tta': 'off' if args.no_tta else 'on',
        'best_ap50': f'{best_ap50:.4f}',
        'best_ap5095': f'{best_ap5095:.4f}',
        'best_epoch': best_epoch,
        'train_time_min': f'{train_time:.1f}',
        'epochs': args.epochs,
        'final_loss': f'{final_loss:.4f}',
    })


# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────

def predict(args):
    if args.gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
            str(g) for g in args.gpu_ids)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\n{"=" * 60}')
    print('  Mode   : PREDICTION')
    print(f'  Device : {device}')
    print(f'  TTA    : {not args.no_tta}')
    print(f'{"=" * 60}\n')

    # Load model using same build_model as training
    ckpt = args.checkpoint
    if ckpt is None:
        exp_dir = Path(args.output_dir) / args.exp_name
        ap50_path = exp_dir / 'best_model.pth'
        if ap50_path.exists():
            ckpt = str(ap50_path)
        else:
            raise FileNotFoundError(f'No checkpoint found in {exp_dir}')

    print(f'Loading checkpoint: {ckpt}')
    model = build_model(args)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.to(device)
    model.eval()

    # Robust filename → image_id mapping
    with open(args.id_json) as f:
        id_list = json.load(f)
    name_to_id = {}
    for item in id_list:
        fname = item['file_name']
        name_to_id[fname] = item['id']
        name_to_id[Path(fname).name] = item['id']
        name_to_id[Path(fname).stem] = item['id']
        name_to_id[Path(fname).stem + '.tif'] = item['id']

    def lookup_id(fpath):
        p = Path(fpath)
        for key in [p.name, p.stem, p.stem + '.tif', str(p)]:
            if key in name_to_id:
                return name_to_id[key]
        return None

    # Run inference
    class TestDataset(Dataset):
        def __init__(self, test_dir):
            self.paths = sorted(Path(test_dir).glob('*.tif'))

        def __len__(self): return len(self.paths)

        def __getitem__(self, idx):
            img_np = _to_rgb(tifffile.imread(str(self.paths[idx])))
            return torch.from_numpy(img_np.transpose(
                2, 0, 1)).float() / 255.0, str(self.paths[idx])

    test_ds = TestDataset(args.test_dir)
    results = []
    skipped = 0
    print(f'Running inference on {len(test_ds)} test images...\n')

    use_tta = not args.no_tta
    for i, (img_tensor, img_path) in enumerate(test_ds, 1):
        image_id = lookup_id(img_path)
        if image_id is None:
            print(f'  [WARN] {Path(img_path).name} not in id_json — skip')
            skipped += 1
            continue

        with torch.no_grad():
            scores, labels, masks, _ = predict_single(
                model, img_tensor, device,
                score_thresh=args.score_thresh, use_tta=use_tta)

        keep = scores >= args.score_thresh
        for sc, lb, m in zip(scores[keep], labels[keep], masks[keep]):
            binary = (m.numpy() > 0.5).astype(np.uint8)
            results.append({'image_id': image_id, 'category_id': int(
                lb), 'segmentation': encode_mask(binary), 'score': float(sc)})

        if i % 10 == 0 or i == len(test_ds):
            n_this = len([r for r in results if r['image_id'] == image_id])
            print(
                f'  [{i:3d}/{len(test_ds)}]'
                f' {Path(img_path).name} \u2192 {n_this} instances'),

    out_dir = Path(args.output_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'test-results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f)

    print(f'\n{"=" * 60}')
    print(f'  Saved: {out_path}')
    print(f'  Total predictions : {len(results)}')
    print(f'  Skipped images    : {skipped}')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    args = get_args()
    if args.mode == 'predict':
        predict(args)
    else:
        train(args)

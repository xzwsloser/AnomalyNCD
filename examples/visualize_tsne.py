"""
任务1（Task1）训练前可视化脚本。

需求：初始化 classifier 为 ETF 模块后，在训练开始之前，将**图像特征**与
**classifier 中的 prototype**用 t-SNE 降维并绘制到同一张图中。

说明：
- 本脚本不进行任何训练，仅加载预训练编码器（DINO ViT）抽取图像特征，
  并读取 ETF 分类头的等角原型矩阵 ori_M，两者一起做 t-SNE 可视化。
- 图像特征指预训练编码器的输出（768 维），原型指 ETF 每个类别的方向向量
  （ori_M 的列向量，形状 (num_classes, feat_dim)）。
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.modules.load_backbone import load_backbone
from models.modules._classifier import ETF_Classifier


def get_args():
    parser = argparse.ArgumentParser(
        description='Task1: 训练前图像特征与 ETF 原型的 t-SNE 可视化',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 数据与采样
    parser.add_argument('--dataset_path', type=str, default='data/mvtec_anomaly_detection',
                        help='MVTec AD 数据集根目录路径')
    parser.add_argument('--samples_per_class', type=int, default=10,
                        help='每个类别采样的图像数量，用于控制 t-SNE 规模')
    parser.add_argument('--prototype_num_classes', type=int, default=None,
                        help='ETF 原型数量（默认与数据集中的类别数一致）')
    parser.add_argument('--seed', type=int, default=3407)

    # 模型与设备
    parser.add_argument('--feat_dim', type=int, default=768, help='编码器特征维度')
    parser.add_argument('--pretrained_backbone', type=str, default='dino_vitb8',
                        help='预训练主干名称（dino_vitb8 / dino_vitb16）')
    parser.add_argument('--mask_layers', type=int, default=9,
                        help='MGViT 掩码引导层的数量（与训练配置保持一致）')

    # 输出
    parser.add_argument('--output', type=str, default='outputs/task1_tsne.png',
                        help='可视化图片保存路径（outputs/ 目录已被 .gitignore 忽略）')

    return parser.parse_args()


def normalize_args(args):
    """补齐演示所需参数。"""
    args.image_size = 224
    args.interpolation = 3
    args.crop_pct = 0.875
    # 保证各目录存在
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    return args



def cls_of(path):
    """从图像路径推导所属类别：.../<class>/test/<defect>/xx.png，向上三层为类别名。"""
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))


def build_transform(args):
    """构造与训练一致的图像预处理（resize + center crop + normalize）。"""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    resize_size = int(args.image_size / args.crop_pct)
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def collect_images(dataset_path, samples_per_class, seed):
    """
    从 MVTec AD 数据集中按类别采样图像。

    MVTec 目录结构：dataset_path/<class>/test/<defect_type>/xx.png
    这里对每个类别在其 test 目录下各缺陷子类做均匀采样，确保类别代表性。
    返回 (class_names, image_paths)。
    """
    rng = random.Random(seed)
    class_names = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith('.')
    ])
    image_paths = []
    for cls in class_names:
        test_dir = os.path.join(dataset_path, cls, 'test')
        if not os.path.isdir(test_dir):
            continue
        # 收集该类别 test 下所有子目录里的图片
        class_imgs = []
        for sub in sorted(os.listdir(test_dir)):
            sub_dir = os.path.join(test_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            for fname in sorted(os.listdir(sub_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    class_imgs.append(os.path.join(sub_dir, fname))
        if not class_imgs:
            continue
        # 均匀采样，保证每类图片数量不超 samples_per_class
        sampled = class_imgs if len(class_imgs) <= samples_per_class \
            else rng.sample(class_imgs, samples_per_class)
        image_paths.extend(sampled)
    return class_names, image_paths


@torch.no_grad()
def extract_features(backbone, image_paths, transform, args, device):
    """加载图像并经预训练主干抽取 CLS 特征，形状 (N, feat_dim)。"""
    features = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)
        # 预训练阶段不使用掩码：传全 1 掩码等效于普通自注意力
        mask = torch.ones(1, img_t.size(2), img_t.size(3), device=device)
        feat = backbone(img_t, mask)  # (1, feat_dim)
        features.append(feat.squeeze(0))
    return torch.stack(features, dim=0)


def main():
    args = get_args()
    args = normalize_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")

    # 1. 加载预训练编码器
    backbone = load_backbone(args.pretrained_backbone, mask_layers=args.mask_layers)
    backbone = backbone.to(device).eval()

    # 2. 构建 ETF 分类头并读取固定原型
    class_names, image_paths = collect_images(args.dataset_path, args.samples_per_class, args.seed)
    num_classes = args.prototype_num_classes
    if num_classes is None:
        num_classes = len(class_names)
    etf = ETF_Classifier(feat_dim=args.feat_dim, num_classes=num_classes).to(device)
    prototypes = etf.prototypes().cpu()  # (num_classes, feat_dim)

    # 3. 采样图像并抽取特征
    if not image_paths:
        raise FileNotFoundError(f"在 {args.dataset_path} 下没有找到图像，请检查数据集路径。")
    transform = build_transform(args)
    features = extract_features(backbone, image_paths, transform, args, device).cpu()
    print(f"采样图像特征：{features.shape}，原型：{prototypes.shape}")

    # 4. 将图像特征与原型合并做 t-SNE，保证两者在同一 2D 空间
    combined = torch.cat([features, prototypes], dim=0).numpy()
    tsne = TSNE(n_components=2, random_state=args.seed, init='pca', perplexity=min(30, combined.shape[0]-1))
    emb = tsne.fit_transform(combined)

    # 5. 绘制单张总览图
    n_feats = features.shape[0]
    feat_emb = emb[:n_feats]
    proto_emb = emb[n_feats:]

    # 记录每个特征点所属类别及该类别颜色
    color_map = {cls: plt.get_cmap('tab20')(i % 20) for i, cls in enumerate(class_names)}

    fig, ax = plt.subplots(figsize=(10, 8))
    # 逐类别着色绘制特征点
    for cls in class_names:
        mask = [cls_of(p) == cls for p in image_paths]
        pts = feat_emb[mask]
        ax.scatter(pts[:, 0], pts[:, 1], c=[color_map[cls]], label=cls, alpha=0.7, s=30)

    # 原型用更大的星形标记
    ax.scatter(proto_emb[:, 0], proto_emb[:, 1], marker='*', c='red', s=300,
               label='ETF prototype', edgecolors='black', linewidths=1)
    for j, xy in enumerate(proto_emb):
        ax.annotate(str(j + 1), xy, textcoords='offset points', xytext=(6, 6), fontsize=8)

    ax.set_title('t-SNE of Pre-trained Image Features & ETF Prototypes (Task1)')
    ax.legend(loc='best', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"可视化图片已保存到：{args.output}")


if __name__ == '__main__':
    main()

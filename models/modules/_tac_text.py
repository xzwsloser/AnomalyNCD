"""
TAC 文本引导（Text Counterpart Construction 与文本分支）模块。

参考论文《Image Clustering with External Guidance》(TAC, ICML 2024 Oral, arXiv:2310.11989)
官方实现：reference/2024-ICML-TAC/ 下的 image_embedding.py / text_embedding.py /
filter_nouns.py / retrieve_text.py / models.py::ClusterHead。

整体为非端到端：CLIP 图像/名词特征、faiss k-means 名词筛选、text counterpart 检索
全部离线完成并冻结，生成的文本对应特征作为 AnomalyNCD 训练的常数输入。
CLIP 与 faiss 仅在离线构建时按需导入，避免文本引导关闭时引入额外依赖。
"""

import csv
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# 与 TAC text_embedding.py 一致的多 prompt 模板，编码名词时取平均。
SIMPLE_IMAGENET_TEMPLATES = (
    lambda c: f"itap of a {c}.",
    lambda c: f"a bad photo of the {c}.",
    lambda c: f"a origami {c}.",
    lambda c: f"a photo of the large {c}.",
    lambda c: f"a {c} in a video game.",
    lambda c: f"art of the {c}.",
    lambda c: f"a photo of the small {c}.",
)


def _resolve_device(device=None):
    """默认使用 CUDA（离线阶段需要 faiss/CLIP 的 GPU 加速）。"""
    if device is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def _load_clip_backend(device=None, pretrained_path=None):
    """加载 CLIP 文本/图像编码器，兼容 openai `clip` 与 `open_clip` 两种实现。

    TAC 文本对应构建依赖 CLIP 预训练模型（ViT-B/32）。优先使用 openai/CLIP
    官方的 `clip` 包；若未安装则回退到 `open_clip`（open_clip_torch）。两者均
    缺失时抛出带安装提示的 RuntimeError，避免在不透明的 ImportError 中崩溃。

    若提供 `pretrained_path`（本地 CLIP 权重文件，如 `ViT-B-32.openai.pt` /
    `ViT-B-32.pt`），则直接从本地文件加载，避免服务器离线时回退到 HF 在线下载
    （对应服务器访问不到 huggingface.co 的报错）。

    返回 (clip_model, preprocess, tokenize, backend)，其中 tokenize 统一为
    `texts -> token Tensor` 的可调用对象，屏蔽不同实现 API 差异。
    """
    device = _resolve_device(device)
    if pretrained_path and not os.path.isfile(pretrained_path):
        raise FileNotFoundError(f"CLIP 本地权重文件不存在：{pretrained_path}")

    try:
        import clip  # openai/CLIP 官方包

        clip_name = pretrained_path if pretrained_path else "ViT-B/32"
        clip_model, preprocess = clip.load(clip_name, device=torch.device(device))
        tokenize = lambda texts: clip.tokenize(texts, truncate=True)
        backend = "clip(local)" if pretrained_path else "clip"
    except ImportError:
        try:
            import open_clip  # open_clip_torch 作为回退实现

            # pretrained 传本地文件路径时不联网；否则才按默认名在线下载。
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained=pretrained_path or "openai")
            clip_model = clip_model.to(device).eval()
            tokenize = lambda texts: open_clip.tokenize(texts)
            backend = "open_clip(local)" if pretrained_path else "open_clip"
        except ImportError:
            raise RuntimeError(
                "未找到 CLIP 实现：缺失 `clip`（openai/CLIP）与 `open_clip`"
                "（open_clip_torch）。请先在服务器安装其一，例如：\n"
                "  pip install git+https://github.com/openai/CLIP.git   # openai clip\n"
                "  pip install open_clip_torch                          # open_clip\n"
                "文本对应特征是 Task4/5 的离线前置步骤，还需 faiss 与 CLIP 权重可访问。"
            ) from None
    clip_model.eval()
    return clip_model, preprocess, tokenize, backend


def collect_image_paths(novel_image_root, base_image_root):
    """按 Dataset_AnomalyNCD 的目录约定收集该 category 下所有子图路径。

    novel：{crop_data_path}/{category}/images/{anomaly}/*.png
    base ：{base_data_path}/images/{anomaly}/.../*.png（AeBAD_crop 等）
    返回顺序与图像特征矩阵 / _image_paths.json 严格对齐，供数据集按 image_path 反查。
    """
    paths = []

    def walk(root):
        if not os.path.isdir(root):
            return
        for dirpath, _, files in os.walk(root):
            for fname in sorted(files):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    paths.append(os.path.join(dirpath, fname))

    # base 子图（每个 category 的训练/测试都会用到，需一并生成文本对应）
    walk(os.path.join(base_image_root, 'images'))
    # novel 子图
    walk(os.path.join(novel_image_root, 'images'))
    return paths


def encode_images(paths, clip_model, clip_preprocess, device=None, batch_size=512):
    """CLIP 编码子图→N×512，对应 TAC image_embedding.py。"""
    device = _resolve_device(device)
    feats = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        imgs = [clip_preprocess(Image.open(p).convert("RGB")) for p in chunk]
        imgs = torch.stack(imgs).to(device)
        with torch.no_grad():
            f = clip_model.encode_image(imgs).float().cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def encode_nouns(clip_model, tokenize, noun_csv, device=None, batch_size=4096):
    """CLIP 编码 WordNet 名词表（7 个 prompt 模板平均），对应 TAC text_embedding.py。"""
    device = _resolve_device(device)
    with open(noun_csv, 'r', encoding='utf-8') as f:
        words = [row['word'] for row in csv.DictReader(f)]
    if len(words) == 0:
        raise ValueError(f"名词表为空：{noun_csv}")

    prompt_embeddings = []
    for index in range(len(SIMPLE_IMAGENET_TEMPLATES)):
        batch_feats = []
        for i in range(0, len(words), batch_size):
            chunk = words[i:i + batch_size]
            prompts = [SIMPLE_IMAGENET_TEMPLATES[index](w) for w in chunk]
            text = tokenize(prompts).to(device)
            with torch.no_grad():
                f = clip_model.encode_text(text).float().cpu().numpy()
            batch_feats.append(f)
        prompt_embeddings.append(np.concatenate(batch_feats, axis=0))
    # 多 prompt 平均
    embeddings = np.stack(prompt_embeddings, axis=0).mean(axis=0)
    return embeddings.astype(np.float32)


def filter_nouns(image_feats, nouns_feats, cluster_num, top_k, device=None):
    """faiss spherical k-means 图像中心 + 反向分类逐中心挑选 topK 判别名词。

    对应 TAC filter_nouns.py：先聚类得到图像语义中心，再把每个名词划分到最近中心，
    每中心取置信度 topK 的名词，构成判别性名词子集 filtered_nouns。
    """
    import faiss

    device = _resolve_device(device)
    image_feats = image_feats.astype(np.float32)
    nouns_feats = nouns_feats.astype(np.float32)
    image_feats = image_feats / np.linalg.norm(image_feats, axis=1, keepdims=True)
    nouns_feats = nouns_feats / np.linalg.norm(nouns_feats, axis=1, keepdims=True)

    cluster_num = int(cluster_num)
    if cluster_num >= len(image_feats):
        cluster_num = max(1, len(image_feats) - 1)

    kmeans = faiss.Kmeans(
        image_feats.shape[1], cluster_num, gpu=torch.cuda.is_available(),
        spherical=True, niter=300, nredo=10,
    )
    kmeans.train(image_feats)
    _, preds = kmeans.index.search(image_feats, 1)
    preds = preds.reshape(-1)

    # 每个聚类中心 = 该簇图像特征均值，并 L2 归一化。
    centers = np.zeros((cluster_num, image_feats.shape[1]), dtype=np.float32)
    for k in range(cluster_num):
        centers[k] = image_feats[preds == k].mean(axis=0)
    centers = F.normalize(torch.from_numpy(centers), dim=1).to(device)
    nouns_t = F.normalize(torch.from_numpy(nouns_feats), dim=1).to(device)

    # 名词反向分类：每个名词划分到与其最相似的图像中心。
    similarity = torch.matmul(centers, nouns_t.T)
    softmax_nouns = torch.softmax(similarity, dim=0)
    class_pred = torch.argmax(softmax_nouns, dim=0).long()

    selected = torch.zeros_like(class_pred, dtype=torch.bool)
    for k in range(cluster_num):
        if (class_pred == k).sum() == 0:
            continue
        class_index = torch.where(class_pred == k)[0]
        confidence = softmax_nouns[k, class_index]
        rank = torch.argsort(confidence, descending=True)
        selected[class_index[rank[:top_k]]] = True

    return selected.to('cpu').bool().numpy()


def retrieve_text(image_feats, selected_nouns, tau=0.005, device=None):
    """softmax(feat·noun^T/tau) @ noun 加权求和并 L2 归一化，对应 TAC retrieve_text.py。

    text_i = normalize(softmax(feats_i @ nouns.T / tau) @ nouns)
    """
    device = _resolve_device(device)
    image_feats = image_feats.astype(np.float32)
    image_feats = image_feats / np.linalg.norm(image_feats, axis=1, keepdims=True)
    selected_nouns = selected_nouns.astype(np.float32)
    selected_nouns = selected_nouns / np.linalg.norm(selected_nouns, axis=1, keepdims=True)

    img_t = torch.from_numpy(image_feats).float().to(device)
    noun_t = torch.from_numpy(selected_nouns).float().to(device)

    similarity = torch.matmul(img_t, noun_t.T)
    similarity = torch.softmax(similarity / tau, dim=1)
    retrieval = similarity @ noun_t
    retrieval = F.normalize(retrieval, dim=1).float()
    return retrieval.cpu().numpy()


def build_or_load(novel_image_root, base_image_root, category, out_root,
                  top_k=5, tau=0.005, cluster_num=None,
                  noun_csv='reference/2024-ICML-TAC/data/WordNetNouns.csv',
                  device=None, pretrained_path=None):
    """为该 category 的全部子图生成文本对应特征并落盘；已存在则直接返回。

    落盘格式（is_distributed 无关，共享 untracked 路径）：
    - {out_root}/{category}.npy                ：N×512 文本特征
    - {out_root}/{category}_image_paths.json   ：与特征行对齐的 image_path 列表

    幂等：两个文件都存在时不再重复计算，避免服务器每次重复跑 CLIP。
    """
    npy_path = os.path.join(out_root, f"{category}.npy")
    json_path = os.path.join(out_root, f"{category}_image_paths.json")
    if os.path.exists(npy_path) and os.path.exists(json_path):
        return npy_path, json_path

    device = _resolve_device(device)
    # 允许通过环境变量 CLIP_CHECKPOINT 指定本地权重，未指定则走在线路径。
    if pretrained_path is None:
        pretrained_path = os.environ.get("CLIP_CHECKPOINT")
    # 服务器离线：自动探测共享 data_store 中预置的 openai JIT 权重，避免
    # open_clip 回退到 huggingface.co 在线下载（服务器无外网 → Network
    # unreachable 挂起直至 KeyboardInterrupt）。CLIP_CHECKPOINT 仍可覆盖。
    if not pretrained_path:
        _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.realpath(__file__))))
        _candidate = os.path.join(_repo_root, "data_store", "clip", "ViT-B-32.pt")
        if os.path.isfile(_candidate):
            pretrained_path = _candidate

    paths = collect_image_paths(novel_image_root, base_image_root)
    if len(paths) == 0:
        raise ValueError(
            f"没有找到该 category 的子图：novel_root={novel_image_root}, "
            f"base_root={base_image_root}"
        )

    # 仅在真正需要生成文本特征时按需加载 CLIP（clip / open_clip 均可）。
    clip_model, clip_preprocess, tokenize, _ = _load_clip_backend(
        device, pretrained_path=pretrained_path)

    # --- Text Counterpart Construction：图像特征 / 名词特征 / 名词筛选 / 文本检索 ---
    image_feats = encode_images(paths, clip_model, clip_preprocess, device)
    nouns_feats = encode_nouns(clip_model, tokenize, noun_csv, device)

    if cluster_num is None or int(cluster_num) <= 0:
        # 默认取该 category 的 novel 缺陷类别数作为图像聚类中心数。
        cluster_num = len(os.listdir(os.path.join(novel_image_root, 'images')))

    selected_idx = filter_nouns(image_feats, nouns_feats, cluster_num, top_k, device)
    if selected_idx.sum() == 0:
        raise ValueError(f"{category} 未筛选到任何判别名词")
    selected_nouns = nouns_feats[selected_idx]

    text_feats = retrieve_text(image_feats, selected_nouns, tau=tau, device=device)

    os.makedirs(out_root, exist_ok=True)
    np.save(npy_path, text_feats.astype(np.float32))
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(paths, f)
    print(f"[build_text_counterpart] {category}: {len(paths)} images, "
          f"selected {int(selected_idx.sum())} nouns -> {npy_path}")
    return npy_path, json_path


class TextProjector(nn.Module):
    """可训练的小文本投影（可选，Task4 拼接前使用）。

    默认关闭；开启时把文本特征线性投影后再与图像 cls_token 拼接，
    让文本空间到可学习特征空间的距离更利于联合分类。
    """

    def __init__(self, text_dim=512):
        super().__init__()
        self.proj = nn.Linear(text_dim, text_dim)

    def forward(self, text_feat):
        return self.proj(text_feat)


class TextClusterHead(nn.Module):
    """TAC ClusterHead 的文本分支，输出跨类分配概率。

    借鉴 TAC models.py::ClusterHead.cluster_head_text：
    Linear(512)->BN1d->ReLU->Linear(512, num_clusters)->Softmax。
    用于 Task5 的 Cross-modal Mutual Distillation：以文本特征给出聚类分配，
    与图像分支的分配互为教师-学生。
    """

    def __init__(self, in_dim=512, num_clusters=10):
        super().__init__()
        self.num_clusters = int(num_clusters)
        self.cluster_head_text = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, self.num_clusters),
            nn.Softmax(dim=1),
        )
        nn.init.trunc_normal_(self.cluster_head_text[0].weight, std=0.02)
        nn.init.trunc_normal_(self.cluster_head_text[3].weight, std=0.02)

    def forward(self, text_feat):
        return self.cluster_head_text(text_feat)

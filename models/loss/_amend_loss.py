import torch
import torch.nn as nn
import torch.nn.functional as F


class AMENDNeighborhoodLoss(nn.Module):
    """AMEND 的直接邻域与扩展邻域对比损失（论文公式 1-3）。

    bank 中的特征全部 detach，仅作为正样本检索来源；负样本只来自当前
    mini-batch，并排除同一原图的两个 view，符合论文的负样本约束。
    """

    def __init__(self, bank_size=2048, neighbors=4, expanded_neighbors=5,
                 expanded_affinity=0.1, temperature=0.1):
        super().__init__()
        if bank_size <= 0:
            raise ValueError("bank_size must be positive")
        if neighbors <= 0 or expanded_neighbors <= 0:
            raise ValueError("neighbors and expanded_neighbors must be positive")
        if neighbors > bank_size or expanded_neighbors > bank_size:
            raise ValueError("neighbor counts cannot exceed bank_size")

        self.bank_size = int(bank_size)
        self.neighbors = int(neighbors)
        self.expanded_neighbors = int(expanded_neighbors)
        self.expanded_affinity = float(expanded_affinity)
        self.temperature = float(temperature)

        self.register_buffer("bank_features", torch.empty(0), persistent=False)
        self.register_buffer("bank_ids", torch.empty(0, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def _topk_neighbors(self, query_features, exclude_ids, topk):
        """从 FIFO bank 检索 top-k 邻居，同时排除指定原图 ID。"""
        if self.bank_features.numel() == 0:
            empty_features = self.bank_features.new_empty(0, 0)
            empty_ids = self.bank_ids.new_empty(0)
            return empty_features, empty_ids

        similarities = query_features @ self.bank_features.T
        exclude = self.bank_ids.unsqueeze(0) == exclude_ids.to(self.bank_ids.device).unsqueeze(1)
        similarities = similarities.masked_fill(exclude, float("-inf"))
        actual_topk = min(int(topk), similarities.shape[1])
        _, indices = similarities.topk(actual_topk, dim=1)

        neighbor_features = self.bank_features[indices]
        neighbor_ids = self.bank_ids[indices]
        return neighbor_features, neighbor_ids

    def _neighborhood_loss(self, anchors, anchor_ids, positive_features):
        """按论文公式 (1)(2) 计算一组正样本的对比损失。

        anchors 已按正样本数量 repeat；分母包含当前正样本和 mini-batch
        negatives，等价于论文中的完整 softmax 分母。
        """
        current_mask = self._current_ids.unsqueeze(0) != anchor_ids.unsqueeze(1)
        if not torch.any(current_mask):
            return anchors.sum() * 0.0

        # current_mask 的行对应 repeat 后的 anchor，列对应当前 batch 样本；
        # 排除项填充 -inf，因此不会进入 softmax 分母。
        negatives = (anchors @ self._current_features.T / self.temperature).masked_fill(
            ~current_mask, float("-inf")
        )
        positives = torch.sum(anchors * positive_features, dim=-1, keepdim=True) / self.temperature
        logits = torch.cat([positives, negatives], dim=1)
        log_prob = positives - torch.logsumexp(logits, dim=1, keepdim=True)
        return -log_prob.mean()

    def forward(self, projections, original_ids, return_components=False):
        if projections.ndim != 2 or original_ids.ndim != 1:
            raise ValueError("projections must be 2-D and original_ids must be 1-D")
        if projections.shape[0] != original_ids.shape[0] or projections.shape[0] % 2 != 0:
            raise ValueError("projections must contain two views for each original sample")

        features = F.normalize(projections, dim=-1, p=2)
        original_ids = original_ids.to(features.device).long()
        if features.shape[0] == 0:
            zero = features.sum() * 0.0
            return (zero, zero, zero) if return_components else zero

        # 首个 batch 还没有历史邻居；仍更新 bank，使下一批可用 AMEND 损失。
        if self.bank_features.numel() == 0:
            self._append_to_bank(features.detach(), original_ids.detach())
            zero = features.sum() * 0.0
            return (zero, zero, zero) if return_components else zero

        self._current_features = features.detach()
        self._current_ids = original_ids.detach()

        anchors = features.repeat_interleave(self.neighbors, dim=0)
        anchor_ids = original_ids.repeat_interleave(self.neighbors, dim=0)
        direct_features, direct_ids = self._topk_neighbors(
            features, original_ids, self.neighbors
        )
        direct_loss = self._neighborhood_loss(
            anchors, anchor_ids, direct_features.reshape(-1, features.shape[1])
        )

        # 每个直接邻居再检索 M 个邻居；不合并二级列表，保留重复项以放大近邻权重。
        expanded_query_ids = direct_ids.reshape(-1)
        expanded_exclude_ids = direct_ids.reshape(-1)
        expanded_features, _ = self._topk_neighbors(
            direct_features.reshape(-1, features.shape[1]), expanded_exclude_ids,
            self.expanded_neighbors,
        )
        expanded_anchors = features.repeat_interleave(
            self.neighbors * self.expanded_neighbors, dim=0
        )
        expanded_anchor_ids = original_ids.repeat_interleave(
            self.neighbors * self.expanded_neighbors, dim=0
        )
        expanded_loss = self._neighborhood_loss(
            expanded_anchors, expanded_anchor_ids, expanded_features.reshape(-1, features.shape[1])
        )

        self._append_to_bank(features.detach(), original_ids.detach())
        self._current_features = None
        self._current_ids = None
        total_loss = direct_loss + self.expanded_affinity * expanded_loss
        if return_components:
            return total_loss, direct_loss, expanded_loss
        return total_loss

    @torch.no_grad()
    def _append_to_bank(self, features, original_ids):
        """以 FIFO 方式写入 l2 normalized projection 特征。"""
        keep = max(0, self.bank_size - features.shape[0])
        if keep == 0:
            self.bank_features = features[-self.bank_size:].clone()
            self.bank_ids = original_ids[-self.bank_size:].clone()
        else:
            self.bank_features = torch.cat([self.bank_features[-keep:], features], dim=0)
            self.bank_ids = torch.cat([self.bank_ids[-keep:], original_ids], dim=0)


class AdaptiveMarginLoss(nn.Module):
    """AMEND 的 adaptive margin 原型正则（论文公式 6-7）。"""

    def forward(self, prototypes):
        if prototypes.ndim != 2:
            raise ValueError("prototypes must have shape (num_classes, feat_dim)")
        if prototypes.shape[0] < 2:
            return prototypes.sum() * 0.0

        # 分类 logits 和 margin 都使用归一化后的原型，符合论文 3.2 节定义。
        prototypes = F.normalize(prototypes, dim=-1, p=2)
        dot_products = prototypes @ prototypes.T
        squared_distances = torch.cdist(prototypes, prototypes, p=2) ** 2
        pair_mask = torch.triu(torch.ones_like(dot_products, dtype=torch.bool), diagonal=1)

        pair_dots = dot_products[pair_mask]
        pair_distances = squared_distances[pair_mask]
        mean_distance = (-pair_distances).mean()
        pair_losses = -pair_distances + mean_distance + pair_dots

        # 论文公式 (6) 外层归一化系数是 1/C，而非 1/有效 pair 数，保持一致。
        return pair_losses.mean() / prototypes.shape[0]

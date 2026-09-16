"""
TAC Cross-modal Mutual Distillation 损失模块（Task5）。

参考论文《Image Clustering with External Guidance》(TAC, ICML 2024 Oral, arXiv:2310.11989)
官方实现：reference/2024-ICML-TAC/loss_utils.py（DistillLoss / consistency_loss / entropy）。

跨模态互蒸馏：拉近「图像分支对某样本的分配」与「文本分支对同一样本（或邻居）的分配」，
互为教师-学生（InfoNCE 式 DistillLoss + 分配一致性 CE + 负熵防退化）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy(logit):
    """负熵正则，防止所有样本退化为同一类（对应 TAC loss_utils.entropy）。"""
    logit = logit.mean(dim=0)
    logit_ = torch.clamp(logit, min=1e-9)
    b = logit_ * torch.log(logit_)
    return -b.sum()


def consistency_loss(anchors, neighbors):
    """对齐两分支的分配分布：让同一样本的文本/图像分配点积趋近 1（binary CE）。

    对应 TAC loss_utils.consistency_loss。
    """
    b, n = anchors.size()
    similarity = torch.bmm(anchors.view(b, 1, n), neighbors.view(b, n, 1)).squeeze()
    ones = torch.ones_like(similarity)
    return F.binary_cross_entropy(similarity, ones)


class TACDistillLoss(nn.Module):
    """InfoNCE 式跨模态蒸馏损失，等价 TAC loss_utils.DistillLoss。

    将两分支的聚类分配按类维度拼接后做对比：同一类别的跨模态分配互为正样本，
    其余为负样本；用 CrossEntropy 拉近互为教师-学生的一对分配。
    """

    def __init__(self, class_num, temperature=0.5):
        super().__init__()
        self.class_num = int(class_num)
        self.temperature = float(temperature)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        # mask 在 forward 中按实际设备惰性构建，避免早期只注册到 CPU 再迁移。
        self._mask = None

    def _build_mask(self, device):
        """构造类间对角线 mask：排除自身与跨模态同类（正样本）以外的项。"""
        N = 2 * self.class_num
        mask = torch.ones((N, N), dtype=torch.bool, device=device)
        mask.fill_diagonal_(0)
        for i in range(self.class_num):
            mask[i, self.class_num + i] = 0
            mask[self.class_num + i, i] = 0
        return mask

    def forward(self, c_i, c_j):
        if self._mask is None or self._mask.device != c_i.device:
            self._mask = self._build_mask(c_i.device)
        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        c = F.normalize(c, dim=1)
        sim = c @ c.T / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self._mask].reshape(N, -1)

        labels = torch.zeros(N, device=c_i.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        return self.criterion(logits, labels) / N

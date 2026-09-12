import numpy as np
import torch
import torch.nn as nn

"""
The code is based on SimGCD and UNO, the source code is available at:
https://github.com/DonkeyShot21/UNO/blob/main/utils/nets.py (UNO)
https://github.com/CVMI-Lab/SimGCD/blob/main/model.py (SimGCD)
"""


class ETF_Classifier(nn.Module):
    """
    等角紧框架（Equiangular Tight Frame, ETF）分类头。

    参考论文《Novel Class Discovery for Long-tailed Recognition》(arXiv:2308.02989)
    及其官方实现 reference/NCDLR/nets/vit.py::ETF_Classifier。
    核心思想：将类别原型固定为等角紧框架（所有原型两两夹角相同、模长相同），
    前向时对输入特征归一化后与等角原型矩阵相乘得到 logits：
        logits = normalize(x) @ ori_M
    其中 ori_M 形状为 (feat_dim, num_classes)。
    """

    def __init__(self, feat_dim, num_classes):
        super(ETF_Classifier, self).__init__()
        assert feat_dim >= num_classes, \
            "ETF 要求特征维度 feat_dim 不小于类别数 num_classes"
        # 生成等角原型矩阵，并用 register_buffer 保证随模型一起移动设备
        ori_M = self._generate_equiangular_matrix(feat_dim, num_classes)
        self.register_buffer("ori_M", ori_M)

    def _generate_equiangular_matrix(self, feat_dim, num_classes):
        """构造等角紧框架矩阵 ori_M，形状 (feat_dim, num_classes)。"""
        # 随机正交矩阵 P（feat_dim x num_classes），满足 P^T P = I
        P = self._generate_random_orthogonal_matrix(feat_dim, num_classes)
        I = torch.eye(num_classes, dtype=P.dtype)
        ones = torch.ones(num_classes, num_classes, dtype=P.dtype)
        # 等角紧框架公式：M = sqrt(C/(C-1)) * P @ (I - (1/C) * ones)，C 为类别数
        scale = np.sqrt(num_classes / (num_classes - 1))
        M = scale * (P @ (I - (1.0 / num_classes) * ones))
        return M

    @staticmethod
    def _generate_random_orthogonal_matrix(feat_dim, num_classes):
        """通过 QR 分解生成随机正交矩阵，满足 P^T P = I。"""
        a = np.random.random((feat_dim, num_classes))
        P, _ = np.linalg.qr(a)
        return torch.from_numpy(P).float()

    def prototypes(self):
        """返回每个类别的原型向量，形状 (num_classes, feat_dim)。"""
        return self.ori_M.T.detach()

    def forward(self, x):
        # 对特征做 L2 归一化，再与等角原型矩阵相乘得到 logits
        x = nn.functional.normalize(x, dim=-1, p=2)
        return x @ self.ori_M


class MultiHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True,
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256, n_head=4,
                 use_etf=False):
        super().__init__()
        self.num_head = n_head
        self.use_etf = use_etf

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        elif nlayers != 0:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

        if use_etf:
            # ETF 模式：使用固定等角原型分类头，所有 head 共享同一个等角矩阵
            self.etf = ETF_Classifier(in_dim, out_dim)
            self.last_layer = None
        else:
            # 默认模式：可学习的 weight-norm 分类头（DINO/SwAV 风格）
            self.last_layer = nn.ModuleList(
                [nn.utils.weight_norm(nn.Linear(in_dim, out_dim, bias=False)) for _ in range(n_head)])
            for i in range(n_head):
                self.last_layer[i].weight_g.data.fill_(1)

            if norm_last_layer:
                for i in range(n_head):
                    self.last_layer[i].weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def prototypes(self):
        """返回 classifier 中的 prototype 向量，形状 (num_classes, feat_dim)。"""
        if self.use_etf:
            return self.etf.prototypes()
        # 非 ETF 模式：取第一个 head 的线性层权重作为原型
        w = self.last_layer[0].weight.detach()
        return w

    def forward(self, x):
        # x_proj：MLP 投影，仅用于对比学习损失
        x_proj = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        if self.use_etf:
            # ETF 模式下多个 head 共享同一个等角矩阵，输出相同 logits
            logits = [self.etf(x) for _ in range(self.num_head)]
        else:
            logits = [self.last_layer[i](x) for i in range(self.num_head)]

        return x_proj, logits


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

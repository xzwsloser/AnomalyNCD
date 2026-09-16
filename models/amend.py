import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets.data_utils import get_pseudo_label_weights
from models.AnomalyNCD import AnomalyNCD
from models.loss._amend_loss import AMENDNeighborhoodLoss, AdaptiveMarginLoss
from models.loss._contrastive_loss import SupConLoss
from models.modules._classifier import MultiHead
from models.modules.load_backbone import load_backbone
from utils.general_utils import AverageMeter


class AMENDProjector(nn.Module):
    """AMEND 专用 projector：MLP 输出用于邻域损失，归一化原型用于分类。"""

    def __init__(self, in_dim, out_dim, projection_dim=256, use_etf=False):
        super().__init__()
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive")

        self.use_etf = use_etf
        self.projector = MultiHead(
            in_dim=in_dim,
            out_dim=out_dim,
            nlayers=3,
            bottleneck_dim=projection_dim,
            n_head=1,
            use_etf=use_etf,
        )

    def forward(self, features):
        x_proj = self.projector.mlp(features)
        normalized_features = F.normalize(features, dim=-1, p=2)
        if self.use_etf:
            logits = [self.projector.etf(normalized_features)]
        else:
            # AMEND 论文要求 hidden feature 与 prototype 都做 l2 normalization。
            prototypes = F.normalize(self.classifier_weight(), dim=-1, p=2)
            logits = [normalized_features @ prototypes.T]
        return x_proj, logits

    def classifier_weight(self):
        # torch 2.0.1 的 weight_norm 存在 bug：模型迁移到 GPU 后 `.weight` 属性仍会
        # 返回 CPU 张量，导致与 cuda 特征做矩阵乘时报 device 不一致。这里直接由
        # weight_g / weight_v 手动重建分类头权重（与 weight_norm 的 reparametrization
        # 定义一致），保证结果张量位于正确设备。
        clf = self.classifier
        return clf.weight_g * F.normalize(clf.weight_v, dim=0)

    @property
    def classifier(self):
        # 动态获取权重引用，避免额外注册引用在旧版 PyTorch 中造成设备迁移歧义。
        return self.projector.last_layer[0]


class AMEND(AnomalyNCD):
    """AnomalyNCD 输入输出协议下的 AMEND 论文方法复现。"""

    def __init__(self, args):
        super().__init__(args)
        self._next_sample_id = 0
        self.neighborhood_criterion = AMENDNeighborhoodLoss(
            bank_size=self.args.amend_bank_size,
            neighbors=self.args.amend_neighbors,
            expanded_neighbors=self.args.amend_expanded_neighbors,
            expanded_affinity=self.args.amend_expanded_affinity,
            temperature=self.args.amend_temperature,
        )
        self.margin_criterion = AdaptiveMarginLoss()

    def load_model(self):
        # 复用 MEBin 输入配套的 mask-guided DINO backbone 和冻结策略。
        backbone = load_backbone(self.args.pretrained_backbone, mask_layers=self.args.mask_layers)
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        for name, parameter in backbone.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
                if block_num >= self.args.grad_from_block:
                    parameter.requires_grad = True

        projector = AMENDProjector(
            in_dim=self.args.feat_dim,
            out_dim=self.args.mlp_out_dim,
            projection_dim=self.args.amend_projection_dim,
            use_etf=self.args.use_etf,
        )
        model = nn.Sequential(backbone, projector).to(self.device)
        # 兼容旧版 PyTorch：显式确保 weight-norm 分类器与 backbone 在同一设备。
        projector.classifier.to(self.device)
        self.neighborhood_criterion.to(self.device)
        return model

    def MGRL(self, epoch, optimizer, cluster_criterion):
        """Mask-guided AMEND training，替换 baseline 的 unsupervised contrastive loss。"""
        loss_record = AverageMeter()
        cluster_loss_for_test = []
        total_loss = 0

        json_path = f"{self.args.crop_data_path}/scores_json/{self.args.category}.json"
        with open(json_path, 'r') as handle:
            anomaly_score_json = json.load(handle)

        self.model.train()
        for batch_idx, batch in enumerate(self.train_loader):
            images, class_labels, image_path, masks, mask_path = batch
            sample_weights, mask_lab = get_pseudo_label_weights(
                image_path, self.args.anomaly_thred, self.args.base_category, anomaly_score_json
            )
            sample_weights = torch.tensor(sample_weights * 2).cuda(non_blocking=True)

            # AnomalyNCD 原协议要求每个 batch 同时包含 labelled 与 unlabeled 样本。
            if torch.any(mask_lab) and not torch.all(mask_lab):
                class_labels = class_labels.cuda(non_blocking=True)
                mask_lab = mask_lab.cuda(non_blocking=True).bool()
                images = torch.cat(images, dim=0).cuda(non_blocking=True)
                masks = torch.cat(masks, dim=0).cuda(non_blocking=True)

                backbone, projector = self.model
                cls_token = backbone(images, masks)
                student_proj, student_out = projector(cls_token)
                teacher_out = [logits.detach() for logits in student_out]

                # 使用连续唯一 ID 标记原图，两个增强 view 共享同一 ID。
                batch_size = class_labels.shape[0]
                original_ids = torch.arange(
                    self._next_sample_id, self._next_sample_id + batch_size,
                    device=student_proj.device, dtype=torch.long,
                ).repeat(2)
                self._next_sample_id += batch_size

                neighborhood_loss, direct_loss, expanded_loss = self.neighborhood_criterion(
                    student_proj, original_ids, return_components=True
                )
                margin_loss = self.margin_criterion(projector.classifier_weight())

                student_proj_lab = torch.cat(
                    [feature[mask_lab].unsqueeze(1) for feature in student_proj.chunk(2)], dim=1
                )
                student_proj_lab = F.normalize(student_proj_lab, dim=-1)
                sup_con_loss = SupConLoss()(student_proj_lab, labels=class_labels[mask_lab])

                student_out_unlabel = torch.cat(
                    [logits[~mask_lab] for logits in (student_out[0]).chunk(2)], dim=0
                )
                teacher_out_unlabel = torch.cat(
                    [logits[~mask_lab] for logits in (teacher_out[0]).chunk(2)], dim=0
                )
                avg_probs = (student_out_unlabel / 0.1).softmax(dim=1).mean(dim=0)
                me_max_loss = -torch.sum(torch.log(avg_probs ** (-avg_probs))) + math.log(float(len(avg_probs)))
                cluster_loss = cluster_criterion(
                    student_out_unlabel, teacher_out_unlabel, epoch, sample_weights
                ) + self.args.memax_weight * me_max_loss
                cluster_loss_for_test.append(cluster_loss.item())

                sup_logits = torch.cat(
                    [logits[mask_lab] for logits in (student_out[0] / 0.1).chunk(2)], dim=0
                )
                sup_labels = torch.cat([class_labels[mask_lab] for _ in range(2)], dim=0)
                cls_loss = nn.CrossEntropyLoss()(sup_logits, sup_labels)

                # 权重完全沿用 AnomalyNCD baseline，保证核心模块差异可比较。
                loss = (1 - self.args.sup_weight) * cluster_loss + self.args.sup_weight * cls_loss
                loss += (1 - self.args.sup_weight) * neighborhood_loss + self.args.sup_weight * sup_con_loss
                loss += self.args.margin_weight * margin_loss

                loss_record.update(loss.item(), batch_size)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if batch_idx % self.args.print_freq == 0:
                    self.args.logger.info(
                        'Epoch: [{}][{}/{}]\t loss {:.5f}\t '
                        'cls_loss: {:.4f} cluster_loss: {:.4f} sup_con_loss: {:.4f} '
                        'direct_loss: {:.4f} expanded_loss: {:.4f} '
                        'neighborhood_loss: {:.4f} adaptive_margin_loss: {:.4f}'.format(
                            epoch, batch_idx, len(self.train_loader), loss.item(),
                            cls_loss.item(), cluster_loss.item(), sup_con_loss.item(),
                            direct_loss.item(), expanded_loss.item(),
                            neighborhood_loss.item(), margin_loss.item(),
                        )
                    )

        cluster_loss_head = [np.mean(cluster_loss_for_test)]
        return cluster_loss_head, loss_record

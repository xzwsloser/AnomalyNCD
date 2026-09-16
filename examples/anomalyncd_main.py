import argparse
import os
import sys
sys.path.append(os.getcwd())
from models.AnomalyNCD import AnomalyNCD
from utils.general_utils import load_yaml

import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description='AnomalyNCD', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # ----------------------
    # dataset setting
    # ----------------------
    parser.add_argument('--dataset', type=str, default='mvtec', help='novel dataset name')
    parser.add_argument('--category', default=None, type=str, help='novel category name')
    parser.add_argument('--dataset_path', type=str, default=None, help='input novel image path')
    parser.add_argument('--anomaly_map_path', type=str, default=None, help='input novel anomaly map path')
    parser.add_argument('--binary_data_path', type=str, default=None, help='output novel binary mask path')
    parser.add_argument('--crop_data_path', type=str, default=None, help='output novel crop data path')
    parser.add_argument('--base_data_path', default=None, type=str, help='input base image path')

    # ----------------------
    # experiment setting
    # ----------------------
    parser.add_argument('--config', type=str, default='./configs/AnomalyNCD.yaml', help='config file path')
    parser.add_argument('--runner_name', default='AnomalyNCD', type=str)
    parser.add_argument('--only_test', type=str, default=None, help='test using the trained checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='path of the trained checkpoint')

    args = parser.parse_args()

    return args


def load_args(cfg, args):
    """
    Load args from the config file
    """
    # ----------------------
    # binariation setting
    # ----------------------
    args.sample_rate = cfg['binarization']['sample_rate']
    args.min_interval_len = cfg['binarization']['min_interval_len']
    args.erode = cfg['binarization']['erode']
    # ----------------------
    # model setting
    # ----------------------
    args.grad_from_block = cfg['models']['grad_from_block']
    args.pretrained_backbone = cfg['models']['pretrained_backbone']
    args.mask_layers = cfg['models']['mask_layers']
    args.n_views = cfg['models']['n_views']
    args.n_head = cfg['models']['n_head']
    args.use_etf = cfg['models']['use_etf']
    # ----------------------
    # training setting
    # ----------------------
    args.batch_size = cfg['training']['batch_size']
    args.num_workers = cfg['training']['num_workers']
    args.lr = cfg['training']['lr']
    args.gamma = cfg['training']['gamma']
    args.momentum = cfg['training']['momentum']
    args.weight_decay = cfg['training']['weight_decay']
    args.epochs = cfg['training']['epochs']
    # ----------------------
    # loss setting
    # ----------------------
    args.sup_weight = cfg['loss']['sup_weight']
    args.memax_weight = cfg['loss']['memax_weight']
    args.anomaly_thred = cfg['loss']['anomaly_thred']
    args.teacher_temp = cfg['loss']['teacher_temp']
    args.warmup_teacher_temp = cfg['loss']['warmup_teacher_temp']
    args.warmup_teacher_temp_epochs = cfg['loss']['warmup_teacher_temp_epochs']
    args.repeat_times = cfg['loss']['repeat_times']
    # Task2：按图像级异常分数筛选 novel 训练集；默认关闭，保持 baseline 可复现。
    train_subset_cfg = cfg.get('train_subset', {})
    args.use_train_subset = bool(train_subset_cfg.get('enabled', False))
    args.train_subset_ratio = float(train_subset_cfg.get('ratio', 0.5))
    args.train_subset_normal_class = str(train_subset_cfg.get('normal_class', 'good'))
    # Task4/5：TAC 文本引导（text counterpart + cross-modal mutual distillation）
    text_cfg = cfg.get('text_counterpart', {})
    args.use_text_feat = bool(text_cfg.get('enabled', False))
    args.text_feat_root = str(text_cfg.get('root', 'data_store/text_counterpart'))
    args.text_top_k = int(text_cfg.get('top_k', 5))
    args.text_tau = float(text_cfg.get('tau', 0.005))
    args.text_cluster_num = text_cfg.get('cluster_num')  # 允许 None 自动推断
    args.concat_text = bool(text_cfg.get('concat_in_features', False))
    args.project_text = bool(text_cfg.get('project_text', False))
    # WordNet 名词表来自 TAC 官方参考实现，随仓库提交。
    args.text_noun_csv = str(cfg.get('text_noun_csv', 'reference/2024-ICML-TAC/data/WordNetNouns.csv'))
    cmd_cfg = cfg.get('cmd', {})
    args.cmd_enabled = bool(cmd_cfg.get('enabled', False))
    args.cmd_weight = float(cmd_cfg.get('weight', 0.1))
    args.cmd_entropy_weight = float(cmd_cfg.get('entropy_weight', 1.0))
    args.cmd_temperature = float(cmd_cfg.get('temperature', 0.5))
    # ----------------------
    # experiment setting
    # ----------------------
    args.seed = cfg['experiment']['seed']
    args.print_freq = cfg['experiment']['print_freq']
    args.table_root = cfg['experiment']['table_root']
    args.exp_name = cfg['experiment']['exp_name']
    args.exp_root = cfg['experiment']['exp_root']

    return args


if __name__ == "__main__":
    args = get_args()
    cfg = load_yaml(args.config)
    args = load_args(cfg, args)
    model = AnomalyNCD(args)
    model.main()

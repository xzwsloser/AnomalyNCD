import numpy as np
from torch.utils.data import Dataset

from datasets.dataset import get_anomalyncd_datasets, subsample_dataset

from copy import deepcopy
import json
import math
import os
import torch


def get_datasets(train_transform, test_transform, args):
    """
    Get datasets for training and testing
    
    Args:
        train_transform: transform to apply to training data
        test_transform: transform to apply to test data
        args: arguments from the command line
    Return: 
        train_dataset: MergedDataset which concatenates labelled and unlabelled
        test_dataset: unlabelled for testing
        unlabelled_train_examples_test: unlabelled for training
    """

    # Get datasets
    datasets = get_anomalyncd_datasets(train_transform=train_transform, test_transform=test_transform,
                            base_path=args.base_data_path,
                            labelled_classes=args.train_classes,
                            unlabelled_classes=args.unlabeled_classes,
                            category=args.category,
                            data_root=args.crop_data_path)
    

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for _, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    if getattr(args, "use_train_subset", False):
        selection_path = os.path.join(args.crop_data_path, "selection_json", f"{args.category}.json")
        before_size = len(datasets['train_unlabelled'])
        datasets['train_unlabelled'] = filter_train_unlabelled_by_selection(
            datasets['train_unlabelled'], selection_path
        )
        selection_summary = load_train_subset_summary(selection_path)
        if hasattr(args, "logger"):
            args.logger.info(
                'Train subset: {} -> {} (test remains {}). Summary: {}'.format(
                    before_size, len(datasets['train_unlabelled']), len(datasets['test']), selection_summary
                )
            )

    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']

    return train_dataset, test_dataset


def get_class_splits(args):
    """
    Get the anomaly class names for the labelled and unlabelled category
    """

    # base_category
    base_path = os.path.join(args.base_data_path, 'images')
    args.train_classes = sorted(os.listdir(base_path))
    base_num = 0
    for base_class in args.train_classes:
        base_class_path = os.path.join(base_path, base_class)
        base_class_files = sorted(os.listdir(base_class_path))
        base_num += len(base_class_files)

    # novel_category
    novel_path = os.path.join(args.crop_data_path, args.category, 'images')
    args.unlabeled_classes = sorted(os.listdir(novel_path))    
    novel_num = 0
    for novel_class in args.unlabeled_classes:
        novel_class_path = os.path.join(novel_path, novel_class)
        novel_class_files = sorted(os.listdir(novel_class_path))
        novel_num += len(novel_class_files)        

    print(f'labeled class: {args.train_classes}')
    print(f'unlabeled class: {args.unlabeled_classes}')

    return args


def select_images_by_anomaly_score(image_scores, ratio, normal_class):
    """
    Select novel training images by image-level anomaly score.

    Task2 约定：缺陷类选择最高分样本，normal_class（如 MVTec 的 good）选择最低分样本。
    相同分数时按文件名升序 tie-break，保证结果可复现。
    """
    if not 0 < ratio <= 1:
        raise ValueError("train subset ratio must be in (0, 1], got {}".format(ratio))

    selected = {}
    stats = {}
    for anomaly_type, score_dict in image_scores.items():
        total = len(score_dict)
        if total == 0:
            selected[anomaly_type] = []
            stats[anomaly_type] = {
                "total": 0,
                "selected": 0,
                "actual_ratio": 0.0,
                "direction": "low" if anomaly_type == normal_class else "high",
                "score_min": None,
                "score_max": None,
            }
            continue

        # max(1, round(total * ratio))：按四舍五入处理奇数样本，同时保证非空类别至少保留一张。
        selected_num = max(1, math.floor(total * ratio + 0.5))
        selected_num = min(selected_num, total)
        ordered = sorted(score_dict.items(), key=lambda item: (-item[1], item[0]))
        if anomaly_type == normal_class:
            selected_items = ordered[-selected_num:]
        else:
            selected_items = ordered[:selected_num]

        selected[anomaly_type] = sorted(item[0] for item in selected_items)
        selected_scores = [item[1] for item in selected_items]
        stats[anomaly_type] = {
            "total": total,
            "selected": selected_num,
            "actual_ratio": selected_num / total,
            "direction": "low" if anomaly_type == normal_class else "high",
            "score_min": min(selected_scores),
            "score_max": max(selected_scores),
        }

    return selected, stats


def save_train_subset_selection(selection_path, selected, stats, ratio, normal_class):
    """
    保存 image-level 筛选清单。该清单只用于过滤训练集，测试集保持全量。
    """
    os.makedirs(os.path.dirname(selection_path), exist_ok=True)
    payload = {
        "ratio": ratio,
        "normal_class": normal_class,
        "score_definition": "max_raw_anomaly_map",
        "selected": selected,
        "stats": stats,
    }
    with open(selection_path, "w") as handle:
        json.dump(payload, handle, indent=2)


def filter_train_unlabelled_by_selection(dataset, selection_path):
    """
    根据 selection JSON 过滤 novel 训练数据；不修改测试数据。
    """
    with open(selection_path, "r") as handle:
        selection = json.load(handle)
    selected = selection["selected"]

    selected_prefixes = {
        anomaly_type: set(image_prefixes) for anomaly_type, image_prefixes in selected.items()
    }
    selected_idxs = []
    for idx, data_tuple in enumerate(dataset.data_to_iterate):
        classname, anomaly_type, image_path, _ = data_tuple
        # crop 文件命名为 "{original_stem}_crop{n}.png"，这里还原为原图 stem 与 selection JSON 对齐。
        image_prefix = os.path.basename(image_path).split("_crop")[0]
        if image_prefix in selected_prefixes.get(anomaly_type, set()):
            selected_idxs.append(idx)

    if not selected_idxs:
        raise ValueError(f"No selected images found for train subset: {selection_path}")
    return subsample_dataset(dataset, selected_idxs)


def load_train_subset_summary(selection_path):
    """
    读取筛选统计摘要，用于训练日志核对。
    """
    with open(selection_path, "r") as handle:
        selection = json.load(handle)
    return selection["stats"]


class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx, image_path, mask, mask_path = self.labelled_dataset[item]

        else:

            img, label, uq_idx, image_path, mask, mask_path = self.unlabelled_dataset[item - len(self.labelled_dataset)]


        return img, label, image_path, mask, mask_path

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)
    

class DistributedWeightedSampler(torch.utils.data.distributed.DistributedSampler):
    """
    sample elements from a given list of indices with given probabilities.
    """
    def __init__(self, dataset, weights, num_samples, num_replicas=None, rank=None,
                 replacement=True, generator=None):
        super(DistributedWeightedSampler, self).__init__(dataset, num_replicas, rank)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.weights = self.weights[self.rank::self.num_replicas]
        self.num_samples = self.num_samples // self.num_replicas

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        rand_tensor =  self.rank + rand_tensor * self.num_replicas
        yield from iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples


def get_pseudo_label_weights(image_path, anomaly_thred, base_category, anomaly_score_json):
    """
    Get the weights for the pseudo labels correction.
    """
    sample_weights = []
    mask_lab = torch.ones(len(image_path))
    # load anomaly score for each sub-image.
    for i, pth in enumerate(image_path):
        if base_category not in pth:
            mask_lab[i] = 0
            if anomaly_thred==-1:
                sample_weights.append(0)
            else:
                ano_type = pth.split('/')[-2]
                filename = pth.split('/')[-1]
                ano_score = anomaly_score_json[ano_type][filename]
                sample_weights.append(max(0.0, anomaly_thred-ano_score))
    return sample_weights, mask_lab

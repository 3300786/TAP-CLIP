import os
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, Dataset
from collections import defaultdict
import random

class RelabeledSubset(Dataset):
    def __init__(self, subset, raw_to_new_label_map):
        self.subset = subset
        self.label_map = raw_to_new_label_map  # ✅ 原始标签 → 新标签（0~n-1）

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, raw_label = self.subset[idx]
        return image, self.label_map[raw_label]


def get_dataloaders(root_dir="data/OfficeHomeDataset_10072016/Real_World",
                    class_names=None,
                    batch_size=32, num_shots=5, preprocess=None):
    """
    - root_dir: 数据目录，指向 Art / Clipart / Real_World 等
    - class_names: 指定加载的类别（必须与目录名一致）
    - num_shots: 每类训练样本数
    """

    # 1. 加载数据
    full_dataset = datasets.ImageFolder(root=root_dir, transform=preprocess)

    # 2. 计算原始 label → 统一新 label 的映射 ✅
    raw_to_new_label_map = {full_dataset.class_to_idx[name]: i for i, name in enumerate(class_names)}

    # 3. 过滤出目标类别
    keep_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label in raw_to_new_label_map]
    filtered_dataset = Subset(full_dataset, keep_indices)

    # 4. 构建 raw_label → indices（注意使用原始 label） ✅
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(filtered_dataset):
        label_to_indices[label].append(idx)

    # 5. Few-shot train / val indices
    # 5. Few-shot train / val indices
    train_indices = []
    if num_shots > 0:
        for label, indices in label_to_indices.items():
            train_indices.extend(random.sample(indices, min(len(indices), num_shots)))
    else:
        print("⚠️ [dataset.py] num_shots=0 → train set will be empty (zero-shot setting)")

    val_indices = []
    for label, indices in label_to_indices.items():
        rest = [i for i in indices if i not in train_indices]
        val_indices.extend(random.sample(rest, min(len(rest), 100)))
    # 6. 构建重编码 Dataset ✅
    train_set = RelabeledSubset(Subset(filtered_dataset, train_indices), raw_to_new_label_map)
    val_set = RelabeledSubset(Subset(filtered_dataset, val_indices), raw_to_new_label_map)
    if num_shots == 0:
        train_loader = None
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    print("🔎 Raw → New Label Map:", raw_to_new_label_map)
    print("✅ Total Classes (Prompt):", len(class_names))
    label_vals = [label for _, label in train_set]
    print("🧪 Train Label Distribution:", sorted(set(label_vals)))

    return train_loader, val_loader

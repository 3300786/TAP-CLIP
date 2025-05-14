# utils/eval_metrics.py

import torch
import torch.nn.functional as F
from collections import defaultdict


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for batch in dataloader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        logits = outputs['logits']
        preds = torch.argmax(logits, dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for t, p in zip(labels, preds):
            per_class_total[t.item()] += 1
            if t == p:
                per_class_correct[t.item()] += 1

    # 全局准确率
    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"🎯 Overall Accuracy: {acc:.2f}%")

    # 每类准确率
    print("📊 Per-Class Accuracy:")
    for cls in sorted(per_class_total.keys()):
        total_c = per_class_total[cls]
        correct_c = per_class_correct[cls]
        acc_c = 100.0 * correct_c / total_c if total_c > 0 else 0.0
        print(f" - Class {cls:2d}: {acc_c:.2f}% ({correct_c}/{total_c})")

    return acc


import torch
import torch.nn.functional as F


def attribution_entropy(attribution_scores):
    """
    attribution_scores: tensor of shape [B, T]
    返回每个样本的 entropy（越小越集中）
    """
    eps = 1e-8
    p = attribution_scores + eps  # 防止 log(0)
    entropy = -(p * torch.log(p)).sum(dim=-1)  # shape: [B]
    return entropy.mean().item()


def attribution_variance(attribution_scores, labels):
    """
    attribution_scores: [B, T]
    labels: [B]
    返回：所有类别的归因方差平均值
    """
    from collections import defaultdict
    label_dict = defaultdict(list)

    for a, l in zip(attribution_scores, labels):
        label_dict[int(l.item())].append(a)

    variances = []
    for group in label_dict.values():
        group_tensor = torch.stack(group)  # [N_c, T]
        var = group_tensor.var(dim=0).mean()  # 平均方差
        variances.append(var.item())

    return sum(variances) / len(variances)

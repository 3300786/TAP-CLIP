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

    acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"🎯 Overall Accuracy: {acc:.2f}%")

    print("📊 Per-Class Accuracy:")
    for cls in sorted(per_class_total.keys()):
        total_c = per_class_total[cls]
        correct_c = per_class_correct[cls]
        acc_c = 100.0 * correct_c / total_c if total_c > 0 else 0.0
        print(f" - Class {cls:2d}: {acc_c:.2f}% ({correct_c}/{total_c})")

    return acc


@torch.no_grad()
def evaluate_per_class_accuracy(model, dataloader, device, class_names=None):
    """返回一个 dict: {class_name/idx: accuracy}，便于训练过程记录和画图"""
    model.eval()
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for batch in dataloader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        logits = outputs['logits']
        preds = torch.argmax(logits, dim=1)

        for t, p in zip(labels, preds):
            per_class_total[t.item()] += 1
            if t == p:
                per_class_correct[t.item()] += 1

    acc_dict = {}
    for cls in sorted(per_class_total.keys()):
        total_c = per_class_total[cls]
        correct_c = per_class_correct[cls]
        acc_c = 100.0 * correct_c / total_c if total_c > 0 else 0.0
        name = class_names[cls] if class_names else str(cls)
        acc_dict[name] = acc_c

    return acc_dict


def attribution_entropy(attribution_scores):
    """计算 token attribution 的 entropy，越小代表越集中"""
    eps = 1e-8
    p = attribution_scores + eps
    entropy = -(p * torch.log(p)).sum(dim=-1)
    return entropy.mean().item()


def attribution_variance(attribution_scores, labels):
    """根据 label 分组计算 attribution 的方差"""
    label_dict = defaultdict(list)
    for a, l in zip(attribution_scores, labels):
        label_dict[int(l.item())].append(a)

    variances = []
    for group in label_dict.values():
        group_tensor = torch.stack(group)
        var = group_tensor.var(dim=0).mean()
        variances.append(var.item())

    return sum(variances) / len(variances) if variances else 0.0

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
    entropy = -(p * torch.log(p)).sum(dim=-1)  # [B]
    return entropy.mean()  # 保持为 tensor，便于反向传播



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

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import save_image

def visualize_attribution_for_class(model, dataloader, target_class, epoch, device="cuda", save_dir="visible results/attribution"):
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    count = 0
    max_samples = 3  # 可视化数量

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 筛选目标类
        mask = labels == target_class
        if mask.sum() == 0:
            continue

        selected_images = images[mask][:max_samples]
        selected_labels = labels[mask][:max_samples]

        with torch.no_grad():
            outputs = model(selected_images, selected_labels)
            if 'attribution' not in outputs:
                print("No attribution scores found in model outputs.")
                return

            attributions = outputs['attribution'].cpu()  # [B, T]
            for idx, (img, attr) in enumerate(zip(selected_images.cpu(), attributions)):
                # 绘图
                fig, ax = plt.subplots(figsize=(10, 2))
                sns.barplot(x=list(range(len(attr))), y=attr.tolist(), ax=ax, palette="rocket")
                ax.set_title(f"Epoch {epoch} | Sample {idx} | Class: {target_class}")
                ax.set_xlabel("Prompt Token Index")
                ax.set_ylabel("Attribution Score")
                plt.tight_layout()

                # 保存归因图
                bar_path = os.path.join(save_dir, f"epoch{epoch}_sample{idx}_bar.png")
                plt.savefig(bar_path)
                plt.close()

                # 保存图像本身
                img_path = os.path.join(save_dir, f"epoch{epoch}_sample{idx}_image.png")
                save_image(img, img_path)

        count += selected_images.size(0)
        if count >= max_samples:
            break

    return f"{count} attribution visualizations saved to: {save_dir}"

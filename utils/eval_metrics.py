import os
import torch
import torch.nn.functional as F
from collections import defaultdict
from torchvision.utils import save_image
import matplotlib.pyplot as plt


@torch.no_grad()
def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for images, labels in dataloader:
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
    model.eval()
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    for images, labels in dataloader:
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
    """
    attribution_scores: [B, prompt_len]
    返回: scalar tensor
    """
    eps = 1e-8
    p = attribution_scores + eps
    entropy = -(p * torch.log(p)).sum(dim=-1)  # [B]
    return entropy.mean()


def attribution_variance(attribution_scores, labels):
    """
    attribution_scores: [B, prompt_len]
    labels: [B]
    返回: scalar tensor（可导）
    """
    label_dict = defaultdict(list)
    for a, l in zip(attribution_scores, labels):
        label_dict[int(l.item())].append(a)

    variances = []
    for group in label_dict.values():
        group_tensor = torch.stack(group, dim=0)  # [N, prompt_len]
        var = group_tensor.var(dim=0).mean()  # scalar tensor
        variances.append(var)

    return torch.stack(variances).mean() if variances else torch.tensor(0.0, device=attribution_scores.device)


@torch.no_grad()
def visualize_attribution_for_class(model, dataloader, target_class, epoch,
                                    device="cuda", save_dir="results/attribution", max_samples=3):
    """
    可视化特定类别的 Prompt Attribution。
    每个样本画一张柱状图 + 保存原图。
    支持 FullModel 输出 attribution: [B, C, prompt_len]
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    count = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        mask = labels == target_class
        if mask.sum() == 0:
            continue

        selected_images = images[mask][:max_samples]
        selected_labels = labels[mask][:max_samples]

        outputs = model(selected_images, selected_labels, return_attribution=True)
        if 'attribution' not in outputs:
            print("No attribution scores found.")
            return

        attributions = outputs['attribution'].cpu()  # [B, C, prompt_len]
        for idx, (img, attr) in enumerate(zip(selected_images.cpu(), attributions)):
            attr_c = attr  # [prompt_len]

            # 绘制柱状图
            plt.figure(figsize=(10, 2))
            plt.bar(range(len(attr_c)), attr_c.tolist(), color='orange')
            plt.title(f"Epoch {epoch} | Sample {idx} | Class {target_class}")
            plt.xlabel("Prompt Token Index")
            plt.ylabel("Attribution Score")
            plt.tight_layout()

            bar_path = os.path.join(save_dir, f"epoch{epoch}_sample{idx}_bar.png")
            plt.savefig(bar_path)
            plt.close()

            # 保存原图像
            img_path = os.path.join(save_dir, f"epoch{epoch}_sample{idx}_image.png")
            save_image(img, img_path)

        count += selected_images.size(0)
        if count >= max_samples:
            break

    print(f"✅ Saved {count} attribution visualizations for class {target_class}")
    return count


@torch.no_grad()
def plot_entropy_distribution(model, dataloader, device, save_path="results/attribution_entropy_dist.png"):
    """
    可视化一批样本的归因 entropy 分布（密度图或直方图）
    """
    model.eval()
    entropies = []

    for images, _ in dataloader:
        images = images.to(device)
        outputs = model(images, return_attribution=True)
        attribution = outputs.get("attribution")  # [B, C, prompt_len]

        if attribution is not None:
            mean_attr = attribution.mean(dim=1)  # [B, prompt_len]
            entropy = attribution_entropy(mean_attr)  # scalar
            entropies.append(entropy.item())

    plt.hist(entropies, bins=20, color="orange")
    plt.title("Attribution Entropy Distribution")
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Entropy distribution saved: {save_path}")

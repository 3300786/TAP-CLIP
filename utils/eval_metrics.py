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
    """
    if attribution_scores.size(0) != labels.size(0):
        raise ValueError(f"Mismatch in batch size: attribution_scores={attribution_scores.shape}, labels={labels.shape}")

    classes = torch.unique(labels)
    variances = []
    for c in classes:
        mask = (labels == c)
        group = attribution_scores[mask]  # [Nc, prompt_len]
        if group.size(0) > 1:
            variances.append(group.var(dim=0).mean())
        else:
            variances.append(torch.tensor(0.0).to(attribution_scores.device))  # 忽略单样本类别

    return torch.stack(variances).mean()

@torch.no_grad()
def visualize_attribution_for_class(model, dataloader, target_class, epoch,
                                    device="cuda", save_dir="results/attribution", max_samples=3):
    """
    可视化特定类别的 Prompt Attribution。
    每个样本画一张柱状图 + 保存原图。
    支持 FullModel 输出 attribution: [B, prompt_len]
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

        attributions = outputs['attribution'].cpu()  # [B, prompt_len]

        for idx, (img, attr) in enumerate(zip(selected_images.cpu(), attributions)):
            attr = attr.tolist()
            mean_attr = sum(attr) / len(attr)
            max_attr = max(attr)
            min_attr = min(attr)

            # 可视化归因变化范围
            plt.figure(figsize=(10, 2))
            bars = plt.bar(range(len(attr)), attr, color='orange')
            for i, bar in enumerate(bars):
                if attr[i] == max_attr:
                    bar.set_color('red')
                elif attr[i] == min_attr:
                    bar.set_color('blue')
            plt.title(f"Epoch {epoch} | Sample {idx} | Class {target_class} | max={max_attr:.3f}, min={min_attr:.3f}")
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

def assert_all_on_gpu(model, *tensors):
    """
    检查模型参数和输入是否都在 GPU 上。否则报错。
    用法：
        assert_all_on_gpu(model, image_tensor, prompt_tensor, ...)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine.")

    # 检查模型参数是否都在 CUDA 上
    for name, param in model.named_parameters():
        if param.device.type != 'cuda':
            raise RuntimeError(f"Model parameter '{name}' is on {param.device}, expected CUDA.")

    # 检查每个输入张量是否在 CUDA 上
    for i, tensor in enumerate(tensors):
        if isinstance(tensor, torch.Tensor):
            if tensor.device.type != 'cuda':
                raise RuntimeError(f"Input tensor at position {i} is on {tensor.device}, expected CUDA.")
        else:
            raise TypeError(f"Argument at position {i} is not a torch.Tensor.")

    print("✅ All model parameters and inputs are on CUDA.")

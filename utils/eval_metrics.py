import os
import torch
import torch.nn.functional as F
from collections import defaultdict
from torchvision.utils import save_image
import matplotlib.pyplot as plt

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            logits = outputs['logits'].detach()
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

def evaluate_per_class_accuracy(model, dataloader, device, class_names=None):
    model.eval()
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            logits = outputs['logits'].detach()
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
    eps = 1e-8
    p = attribution_scores + eps
    entropy = -(p * torch.log(p)).sum(dim=-1)
    return entropy.mean()

def attribution_variance(attribution_scores, labels):
    if attribution_scores.size(0) != labels.size(0):
        raise ValueError(f"Mismatch in batch size: attribution_scores={attribution_scores.shape}, labels={labels.shape}")

    classes = torch.unique(labels)
    variances = []
    for c in classes:
        mask = (labels == c)
        group = attribution_scores[mask]
        if group.size(0) > 1:
            variances.append(group.var(dim=0).mean())
        else:
            variances.append(torch.tensor(0.0).to(attribution_scores.device))

    return torch.stack(variances).mean()

def visualize_attribution_for_class(model, dataloader, target_class, epoch,
                                    device="cuda", save_dir="results/attribution", max_samples=3):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    count = 0

    with torch.no_grad():
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

            attributions = outputs['attribution'].cpu()

            for idx, (img, attr) in enumerate(zip(selected_images.cpu(), attributions)):
                attr = attr.tolist()
                max_attr = max(attr)
                min_attr = min(attr)

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
                plt.close('all')

                img_path = os.path.join(save_dir, f"epoch{epoch}_sample{idx}_image.png")
                save_image(img, img_path)

            count += selected_images.size(0)
            if count >= max_samples:
                break

    print(f"✅ Saved {count} attribution visualizations for class {target_class}")
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return count

def plot_entropy_distribution(model, dataloader, device, save_path="results/attribution_entropy_dist.png"):
    model.eval()
    entropies = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            outputs = model(images, return_attribution=True)
            attribution = outputs.get("attribution")

            if attribution is not None:
                mean_attr = attribution.mean(dim=1)
                entropy = attribution_entropy(mean_attr.detach())
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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this machine.")

    for name, param in model.named_parameters():
        if param.device.type != 'cuda':
            raise RuntimeError(f"Model parameter '{name}' is on {param.device}, expected CUDA.")

    for i, tensor in enumerate(tensors):
        if isinstance(tensor, torch.Tensor):
            if tensor.device.type != 'cuda':
                raise RuntimeError(f"Input tensor at position {i} is on {tensor.device}, expected CUDA.")
        else:
            raise TypeError(f"Argument at position {i} is not a torch.Tensor.")

    print("✅ All model parameters and inputs are on CUDA.")

# train.py

import torch
from torch.utils.data import DataLoader
from models.model_wrapper import FullModel
from models.clip_wrapper import CLIPWrapper
from dataset import get_dataloaders
from utils.eval_metrics import evaluate_accuracy, evaluate_per_class_accuracy, visualize_attribution_for_class
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import os
from datetime import datetime
import random
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    # ===================== 配置 =====================
    version = "version3"
    tag = datetime.now().strftime("%Y%m%d_%H%M")
    base_dir = f"results/{version}/{tag}"

    device = "cuda"
    prompt_len = 5
    attr_lambda = 0.05
    stab_lambda = 0.1
    epochs = 200
    patience = 10
    lr = 2e-3
    decay = 0.01
    warmup_epoch = 20
    seed = 42
    resume = False
    class_names = ["Backpack", "Alarm_Clock", "Laptop", "Pen", "Mug"]
    pretrained_path = "G:/dsCLIP/open_clip_pytorch_model.bin"

    os.makedirs(base_dir, exist_ok=True)
    log_dir = os.path.join(base_dir, "logs")
    fig_dir = os.path.join(base_dir, "figures")
    model_dir = os.path.join(base_dir, "models")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    set_seed(seed)

    # ===================== 模型初始化 =====================
    clip_model = CLIPWrapper(pretrained_path=pretrained_path, device=device)
    model = FullModel(
        class_names=class_names,
        clip_wrapper=clip_model,
        prompt_len=prompt_len,
        attr_lambda=attr_lambda,
        stab_lambda=stab_lambda,
        adjustor_method='residual',
        class_specific=True,
        warmup_epoch=warmup_epoch
    ).to(device)

    # ===================== Logging =====================
    log_file = os.path.join(log_dir, f"train_{tag}.log")
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    optimizer = torch.optim.AdamW(
        list(model.prompt_learner.parameters()) + list(model.prompt_adjustor.parameters()),
        lr=lr, weight_decay=decay
    )

    start_epoch = 1
    best_acc = 0.0
    best_model_state = model.state_dict()

    checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pt")
    if resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        logging.info(f"✅ Resumed training from epoch {start_epoch}")
    else:
        logging.info("🚀 Starting new training from scratch")

    # ===================== 数据加载 =====================
    train_loader, val_loader = get_dataloaders(
        root_dir="data/OfficeHomeDataset_10072016/Real World",
        class_names=class_names,
        batch_size=32,
        num_shots=5,
        preprocess=clip_model.get_preprocess()
    )

    acc_list = []
    per_class_dict = {cls: [] for cls in class_names}
    entropy_list = []
    variance_list = []
    current = 0

    # ===================== 训练循环 =====================
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        model.training_epoch = epoch
        total_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", ncols=100)
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, labels)
            loss = outputs['loss']
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix(loss=loss.item())

            # Logging entropy loss
            if "loss_entropy" in outputs:
                entropy_val = outputs["loss_entropy"].item()
                variance_val = outputs["loss_variance"].item()
                entropy_list.append(entropy_val)
                variance_list.append(variance_val)
                logging.info(f"[Epoch {epoch}] 🔍 Entropy Loss: {entropy_val:.4f}")
            else:
                entropy_list.append(0.0)
                variance_list.append(0.0)

        avg_loss = total_loss / len(train_loader)
        logging.info(f"[Epoch {epoch}] 🏋️ Avg Train Loss: {avg_loss:.4f}")

        # ===================== 验证 =====================
        acc = evaluate_accuracy(model, val_loader, device)
        acc_list.append(acc)
        logging.info(f"[Epoch {epoch}] 🧪 Val Accuracy: {acc:.2f}%")

        per_cls_acc = evaluate_per_class_accuracy(model, val_loader, device, class_names)
        for cls in class_names:
            per_class_dict[cls].append(per_cls_acc[cls])
        logging.info(f"[Epoch {epoch}] 📊 Per-Class Accuracy: {per_cls_acc}")

        if epoch % 10 == 0:
            _ = visualize_attribution_for_class(model, val_loader, target_class=0, epoch=epoch, device=device)

        # ===================== Early Stopping & Checkpoint =====================
        if acc > best_acc:
            best_acc = acc
            current = 0
            best_model_state = model.state_dict()
            logging.info(f"✅ New best at epoch {epoch}, acc={best_acc:.2f}%")
        else:
            current += 1
            if current == patience:
                logging.info(f"⏹ Early stopping triggered at epoch {epoch}. Best Acc = {best_acc:.2f}%")
                break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }, checkpoint_path)

    # ===================== 保存模型 =====================
    model_path = os.path.join(model_dir, f"best_model_attr_{tag}_acc{best_acc:.2f}.pt")
    torch.save(best_model_state, model_path)
    logging.info(f"📦 Best model saved to: {model_path}")

    # ===================== 保存图像 =====================
    plt.figure(figsize=(10, 6))
    plt.plot(acc_list, label="Total Accuracy", linewidth=2)
    for cls in class_names:
        plt.plot(per_class_dict[cls], label=cls)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy per Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"epoch_acc_curve_acc{best_acc}.png"))
    logging.info("📊 Accuracy plot saved.")

    if entropy_list or variance_list:
        plt.figure(figsize=(8, 4))

        def smooth_curve(values, window=5):
            return np.convolve(values, np.ones(window) / window, mode='valid')
        plt.plot(smooth_curve(entropy_list), label="Entropy Loss (smoothed)", color="orange")
        plt.plot(smooth_curve(variance_list), label="Variance Loss (smoothed)", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Entropy")
        plt.title("Attribution Entropy and variance per Epoch")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "epoch_entropy_and_variance_curve.png"))
        logging.info("📉 Entropy and variance plot saved.")

if __name__ == "__main__":
    train()

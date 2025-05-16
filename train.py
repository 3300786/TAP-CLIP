import torch
from torch.utils.data import DataLoader
from models.model_wrapper import FullModel
from models.clip_wrapper import CLIPWrapper
from dataset import get_dataloaders
from utils.eval_metrics import evaluate_accuracy, evaluate_per_class_accuracy
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import os
from datetime import datetime

def generate_output_paths(version: str):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"results/{version}_{now}"
    paths = {
        "base": base_dir,
        "model_dir": os.path.join(base_dir, "models"),
        "plot_dir": os.path.join(base_dir, "plots"),
        "csv_dir": os.path.join(base_dir, "csv"),
        "log_dir": os.path.join(base_dir, "logs"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

def train():
    # ✅ 配置
    version = "main"
    device = "cuda"
    prompt_len = 5
    attr_lambda = 1.0
    stab_lambda = 0.1
    epochs = 100
    patience = 10
    lr = 2e-3
    decay = 0.01
    class_names = ["Backpack", "Alarm_Clock", "Laptop", "Pen", "Mug"]
    pretrained_path = "G:/dsCLIP/open_clip_pytorch_model.bin"
    output_paths = generate_output_paths(version)

    # ✅ 日志
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(output_paths["log_dir"], f"{version}_train.log")),
            logging.StreamHandler()
        ]
    )

    # ✅ 模型初始化
    clip_model = CLIPWrapper(pretrained_path=pretrained_path, device=device)
    model = FullModel(
        class_names=class_names,
        clip_wrapper=clip_model,
        prompt_len=prompt_len,
        attr_lambda=attr_lambda,
        stab_lambda=stab_lambda,
        adjustor_method='scale',
        class_specific=True
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.prompt_learner.parameters(), lr=lr, weight_decay=decay
    )

    logging.info("🔧 Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(f" - {name} | shape: {tuple(param.shape)}")

    # ✅ 数据加载
    train_loader, val_loader = get_dataloaders(
        root_dir="data/OfficeHomeDataset_10072016/Real World",
        class_names=class_names,
        batch_size=32,
        num_shots=5,
        preprocess=clip_model.get_preprocess()
    )

    best_acc = 0.0
    current = 0
    best_model_state = model.state_dict()

    acc_list = []
    per_class_dict = {cls: [] for cls in class_names}

    for epoch in range(1, epochs + 1):
        model.train()
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

        avg_loss = total_loss / len(train_loader)
        logging.info(f"[Epoch {epoch}] 🏋️ Avg Train Loss: {avg_loss:.4f}")

        # ✅ 验证
        acc = evaluate_accuracy(model, val_loader, device)
        acc_list.append(acc)
        logging.info(f"[Epoch {epoch}] 🧪 Val Accuracy: {acc:.2f}%")

        per_cls_acc = evaluate_per_class_accuracy(model, val_loader, device, class_names)
        for cls in class_names:
            per_class_dict[cls].append(per_cls_acc[cls])
        logging.info(f"[Epoch {epoch}] 📊 Per-Class Accuracy: {per_cls_acc}")

        if acc > best_acc:
            best_acc = acc
            current = 0
            best_model_state = model.state_dict()
        else:
            current += 1
            if current == patience:
                break

    # ✅ 保存最佳模型
    model_path = os.path.join(output_paths["model_dir"], f"best_model_{version}_acc{best_acc:.2f}.pt")
    torch.save(best_model_state, model_path)
    logging.info(f"📦 Model saved: {model_path}")

    # ✅ 绘图
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
    plot_path = os.path.join(output_paths["plot_dir"], f"{version}_acc_curve_acc{best_acc:.2f}.png")
    plt.savefig(plot_path)
    logging.info(f"📊 Accuracy plot saved to {plot_path}")


if __name__ == "__main__":
    train()

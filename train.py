import torch
from torch.utils.data import DataLoader
from models.model_wrapper import FullModel
from models.clip_wrapper import CLIPWrapper
from dataset import get_dataloaders
from utils.eval_metrics import evaluate_accuracy
from tqdm import tqdm
import logging


def train():
    # 参数配置
    device = "cuda"
    prompt_len = 5
    attr_lambda = 1.0
    stab_lambda = 0.1
    epochs = 100
    patience = 10
    lr = 2e-3
    decay = 0.01
    # class_names = ['cat', 'dog', 'car', 'plane']  # 示例
    class_names = ["Backpack", "Alarm_Clock", "Laptop", "Pen", "Mug"]  # 举例

    pretrained_path = "G:/dsCLIP/open_clip_pytorch_model.bin"

    # 模型初始化
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
    # 配置 logging
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S"
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.prompt_learner.parameters(), lr=lr, weight_decay=decay
    )
    print("\n🔧 Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" - {name} | shape: {tuple(param.shape)}")

    # 数据加载
    from dataset import get_dataloaders

    class_names = ["Backpack", "Alarm_Clock", "Laptop", "Pen", "Mug"]  # 举例
    # class_names = ['cat', 'dog', 'car', 'plane']  # 示例

    train_loader, val_loader = get_dataloaders(
        root_dir="data/OfficeHomeDataset_10072016/Real World",
        class_names=class_names,
        batch_size=32,
        num_shots=5,
        preprocess=clip_model.get_preprocess()
    )
    best_acc = 0.0
    current = 0
    # 主训练循环
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

        # 评估
        acc = evaluate_accuracy(model, val_loader, device)
        logging.info(f"[Epoch {epoch}] 🧪 Val Accuracy: {acc:.2f}%")
        # best val?test
        if acc > best_acc:
            best_acc = acc
            current = 0
            if acc > 90:
                model_path = f"Best Models/best_model_epoch{epoch}_acc{acc:.2f}.pt"
                torch.save(model.state_dict(), model_path)
                logging.info(f"📦 Model saved: {model_path}")
        else:
            current += 1
            if current == patience:
                break


if __name__ == "__main__":
    train()


"""
domain shift: allowed for mix bet train sets and val sets (by domain
测试集必须是不同域 domainnet
imagenet-X: train on o-imagenet, test on A/R/S, same as 0/few-shot
domain shift: -unseen domain-, <unseen classes>,
compare with CoOp/CoCoOp
domainnet 40G
OfficeHome 1G
ImageNet-A/R/S several G


CLIP Loss update(image-text, text-image)
attribution: cross-attn, mask(complexity, could be visible part), combine direction by image(additional net, auxi through image feature)

Purpose: Zero/Few-shot
0-<16>

"""
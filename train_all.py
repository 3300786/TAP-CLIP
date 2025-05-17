# train.py

import os
import argparse
import torch
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader
from models.model_wrapper import FullModel
from models.clip_wrapper import CLIPWrapper
from utils.eval_metrics import evaluate_accuracy, evaluate_per_class_accuracy, visualize_attribution_for_class, assert_all_on_gpu
from datasets import OfficeHome, fewshot_sampler


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_run(model, train_loader, val_loader, class_names, optimizer, args, save_dir, device):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "train.log")
    fig_dir = os.path.join(save_dir, "figures")
    model_dir = os.path.join(save_dir, "models")
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    for d in [fig_dir, model_dir, checkpoint_dir]:
        os.makedirs(d, exist_ok=True)

    best_acc = 0.0
    best_state = model.state_dict()
    current = 0
    entropy_list, variance_list, acc_list = [], [], []
    per_class_dict = {cls: [] for cls in class_names}

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.training_epoch = epoch
        total_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100)

        for images, labels in bar:
            images, labels = images.to(device), labels.to(device)
            output = model(images, labels)
            loss = output['loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            bar.set_postfix(loss=loss.item())
            # 在主训练函数中验证
            # assert_all_on_gpu(model, images)

            if "loss_entropy" in output:
                entropy_list.append(output["loss_entropy"].item())
                variance_list.append(output["loss_variance"].item())

        avg_loss = total_loss / len(train_loader)
        acc = evaluate_accuracy(model, val_loader, device)
        acc_list.append(acc)
        print(f"[Epoch {epoch}] Loss={avg_loss:.4f} | Val Acc={acc:.2f}")

        per_cls = evaluate_per_class_accuracy(model, val_loader, device, class_names)
        for cls in class_names:
            per_class_dict[cls].append(per_cls[cls])

        if epoch % 10 == 0:
            visualize_attribution_for_class(model, val_loader, target_class=0, epoch=epoch, device=device)

        if acc > best_acc:
            best_acc = acc
            current = 0
            best_state = model.state_dict()
        else:
            current += 1
            if current == args.patience:
                print(f"⏹ Early stopping at epoch {epoch}. Best Acc = {best_acc:.2f}%")
                break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }, os.path.join(checkpoint_dir, "last_checkpoint.pt"))

    # 保存模型
    best_model_path = os.path.join(model_dir, f"best_model_acc{best_acc:.2f}.pt")
    torch.save(best_state, best_model_path)
    print(f"✅ Best model saved: {best_model_path}")

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(acc_list, label="Total Accuracy", linewidth=2)
    for cls in class_names[:10]:
        plt.plot(per_class_dict[cls], label=cls)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy per Epoch")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"acc_curve.png"))

    if entropy_list:
        def smooth(values, window=5):
            return np.convolve(values, np.ones(window) / window, mode='valid')
        plt.figure(figsize=(8, 4))
        plt.plot(smooth(entropy_list), label="Entropy", color="orange")
        plt.plot(smooth(variance_list), label="Variance", color="blue")
        plt.title("Attribution Regularization")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"entropy_variance_curve.png"))


def main(args):
    domains = ["A", "C", "P", "R"]
    domain_map = {"A": "Art", "C": "Clipart", "P": "Product", "R": "Real_World"}
    dataset = OfficeHome(root="data/")
    class_names = dataset[0].classes  # 自动获取全部类别名

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥 Running on device: {device}")

    for test_id, domain in enumerate(domains):
        for seed in args.seeds:
            set_seed(seed)
            print(f"\n🔁 Domain: {domain_map[domain]} | Seed: {seed}")

            clip = CLIPWrapper(pretrained_path=args.pretrained_path, device=device)
            preprocess = clip.get_preprocess()

            train_sets = []
            for i in range(len(domains)):
                if i != test_id:
                    dataset[i].transform = preprocess
                    few = fewshot_sampler(dataset[i], args.num_shots)
                    train_sets.append(few)
            val_set = dataset[test_id]
            val_set.transform = preprocess

            train_loader = DataLoader(ConcatDataset(train_sets), batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

            model = FullModel(
                class_names=class_names,
                clip_wrapper=clip,
                prompt_len=args.prompt_len,
                prefix_len=args.prefix_len,
                attr_lambda=args.attr_lambda,
                stab_lambda=args.stab_lambda,
                adjustor_method='residual',
                class_specific=True,
                warmup_epoch=args.warmup_epoch
            ).to(device)

            optimizer = torch.optim.AdamW(
                list(model.prompt_learner.parameters()) + list(model.prompt_adjustor.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay
            )

            save_dir = os.path.join("results", args.version, f"{domains[test_id]}", f"seed{seed}")
            train_one_run(model, train_loader, val_loader, class_names, optimizer, args, save_dir, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LODO Training for TAP-CLIP")

    parser.add_argument('--version', type=str, default="version3", help="Experiment version name")
    parser.add_argument('--pretrained_path', type=str, default="G:/dsCLIP/open_clip_pytorch_model.bin")
    parser.add_argument('--prompt_len', type=int, default=3)
    parser.add_argument('--attr_lambda', type=float, default=0.05)
    parser.add_argument('--stab_lambda', type=float, default=0.1)
    parser.add_argument('--warmup_epoch', type=int, default=20)
    parser.add_argument('--num_shots', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--prefix_len', type=int, default=5)

    args = parser.parse_args()
    main(args)

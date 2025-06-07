import os
import argparse
import torch
import random
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import ConcatDataset, DataLoader
from models.model_wrapper import FullModel
from models.clip_wrapper import CLIPWrapper
from utils.eval_metrics import evaluate_accuracy, evaluate_per_class_accuracy, visualize_attribution_for_class
from datasets import OfficeHome, fewshot_sampler, DomainNet
import gc


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_run(model, train_loader, val_loader, class_names, optimizer, args, save_dir, device, scaler, scheduler):
    os.makedirs(save_dir, exist_ok=True)
    model_dir = os.path.join(save_dir, "models")
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    for d in [model_dir, checkpoint_dir]:
        os.makedirs(d, exist_ok=True)

    best_acc = 0.0
    best_state = None
    current = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.training_epoch = epoch
        total_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=100, dynamic_ncols=True)

        for images, labels in bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                output = model(images, labels)
                loss = output['loss']

            if not torch.isfinite(loss).all():
                print(f"⚠️ Skipped step due to NaN/inf loss: {loss.item()}")
                continue

            # ✅ 注释掉日志提取（当前不需要记录）
            # loss_entropy_val = output.get("loss_entropy", torch.tensor(0.0)).detach().item()
            # loss_variance_val = output.get("loss_variance", torch.tensor(0.0)).detach().item()

            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            bar.set_postfix(loss=loss.item())

            # ✅ 注释掉记录列表（当前不需要记录曲线）
            # entropy_list.append(loss_entropy_val)
            # variance_list.append(loss_variance_val)

            del output, images, labels, loss, scaled_loss
            gc.collect()
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_loader)

        with torch.no_grad():
            acc = evaluate_accuracy(model, val_loader, device)
            # acc_list.append(acc)
            # per_cls = evaluate_per_class_accuracy(model, val_loader, device, class_names)
        scheduler.step(acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch}] Loss={avg_loss:.4f} | Val Acc={acc:.2f} | lr={current_lr:.3e}")

        # ✅ 注释掉 per-class 曲线记录
        # for cls in class_names:
        #     per_class_dict[cls].append(per_cls[cls])

        # ✅ 注释掉 attribution 可视化
        # if epoch % 10 == 0:
        #     with torch.no_grad():
        #         visualize_attribution_for_class(model, val_loader, target_class=0, epoch=epoch, device=device)

        if acc > best_acc:
            best_acc = acc
            current = 0
            if acc > 90:
                torch.save(model.state_dict(), os.path.join(model_dir, f"model_epoch{epoch}.pt"))
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
    
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    del obj
            except:
                pass
        gc.collect()
        torch.cuda.empty_cache()

        print(f"[GPU] Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"[GPU] Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    # ✅ 注释掉最终 best_model 保存图
    best_model_path = os.path.join(model_dir, f"best_model_acc{best_acc:.2f}.pt")
    torch.save(best_state, best_model_path)
    print(f"✅ Best model saved: {best_model_path}")

    # ✅ 注释掉绘图部分
    # plt.figure(figsize=(10, 6))
    # plt.plot(acc_list, label="Total Accuracy", linewidth=2)
    # for cls in class_names[:10]:
    #     plt.plot(per_class_dict[cls], label=cls)
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy (%)")
    # plt.title("Validation Accuracy per Epoch")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, "figures", f"acc_curve.png"))

    # if entropy_list:
    #     def smooth(values, window=5):
    #         return np.convolve(values, np.ones(window) / window, mode='valid')
    #     plt.figure(figsize=(8, 4))
    #     plt.plot(smooth(entropy_list), label="Entropy", color="orange")
    #     plt.plot(smooth(variance_list), label="Variance", color="blue")
    #     plt.title("Attribution Regularization")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(save_dir, "figures", f"entropy_variance_curve.png"))


def main(args):
    domains = ["A", "C", "P", "R"]
    domain_map = {"A": "Art", "C": "Clipart", "P": "Product", "R": "Real_World"}
    dataset = OfficeHome(root="data/")
    # dataset = DomainNet(root="/root/autodl-tmp/data/")
    class_names = dataset[0].classes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥 Running on device: {device}")

    for test_id, domain in enumerate(domains):
        for seed in args.seeds:
            set_seed(seed)
            print(f"\n🔁 Domain: {domain_map[domain]} | Seed: {seed}")

            clip = CLIPWrapper(pretrained_path=args.pretrained_path, device=device, lora_layers=args.lora_layers)
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
                # class_specific=True,
                warmup_epoch=args.warmup_epoch,
            ).to(device)

            prompt_params = (
                    list(model.prompt_learner.parameters()) +
                    list(model.prompt_adjustor.parameters())
            )
            for n, p in model.named_parameters():
                if "prompt_learner" in n:
                    print(n, p.shape, p.requires_grad)

            prompt_params = [p for n, p in model.named_parameters()
                             if ("prompt_learner" in n or "prompt_adjustor" in n) and p.requires_grad]

            lora_params = [p for n, p in model.named_parameters()
                           if "lora_A" in n or "lora_B" in n]

            print("Prompt params:", len(prompt_params), "LoRA params:", len(lora_params))

            # ③ 构建分组优化器
            optimizer = torch.optim.AdamW(
                [
                    {"params": prompt_params, "lr": args.lr},  # 2e-3
                    {"params": lora_params, "lr": args.lora_lr},  # 5e-4
                ],
                weight_decay=args.weight_decay
            )

            print("Group-A (prompt)  params:", sum(p.numel() for p in prompt_params))
            print("Group-B (LoRA)    params:", sum(p.numel() for p in lora_params))
            for i, g in enumerate(optimizer.param_groups):
                print(f"group {i}  lr = {g['lr']:.1e},  num_params = {len(g['params'])}")

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3, threshold=0.001, verbose=True
            )
            scaler = GradScaler()
            save_dir = os.path.join("results", args.version, f"{domains[test_id]}", f"seed{seed}")
            train_one_run(model, train_loader, val_loader, class_names, optimizer, args, save_dir, device, scaler, scheduler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LODO Training for TAP-CLIP")

    parser.add_argument('--version', type=str, default="version3", help="Experiment version name")
    parser.add_argument('--pretrained_path', type=str, default="open_clip_pytorch_model_16.bin")
    parser.add_argument('--prompt_len', type=int, default=10)
    parser.add_argument('--attr_lambda', type=float, default=0.01)
    parser.add_argument('--stab_lambda', type=float, default=0.03)
    parser.add_argument('--warmup_epoch', type=int, default=20)
    parser.add_argument('--num_shots', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--prefix_len', type=int, default=5)
    parser.add_argument("--lora_lr", type=float, default=5e-4,
                        help="learning-rate for LoRA layers")
    parser.add_argument("--lora_layers", type=int, default=16,
                        help="LoRA layers")

    args = parser.parse_args()
    main(args)
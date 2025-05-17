# test.py

import os
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from models.model_wrapper import FullModel
from models.clip_wrapper import CLIPWrapper
from utils.eval_metrics import evaluate_accuracy
from datasets import OfficeHome, fewshot_sampler
from torch.utils.data import DataLoader, ConcatDataset


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze_model_except_prompt(model):
    for name, param in model.named_parameters():
        param.requires_grad = "prompt_learner.context_bank" in name


def fine_tune(model, loader, device, steps=10, lr=5e-3):
    freeze_model_except_prompt(model)
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for _ in range(steps):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss = model(images, labels)["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def find_model_path(base_dir, domain, seed):
    model_dir = os.path.join(base_dir, domain, f"seed{seed}", "models")
    for fname in os.listdir(model_dir):
        if fname.startswith("best_model") and fname.endswith(".pt"):
            return os.path.join(model_dir, fname)
    raise FileNotFoundError(f"No model found in {model_dir}")


def evaluate_on_domain(model, val_loader, device):
    model.eval()
    return evaluate_accuracy(model, val_loader, device)


def main(args):
    # 参数准备
    domains = ["A", "C", "P", "R"]
    domain_names = {"A": "Art", "C": "Clipart", "P": "Product", "R": "Real_World"}
    dataset = OfficeHome(root="data/")
    class_names = dataset[0].classes  # 自动获取全部类别名
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥 Running on device: {device}")
    version = args.version
    root_results = f"results/{version}"
    output_dir = f"results/{version}_test/{datetime.now().strftime('%Y%m%d_%H%M')}"
    os.makedirs(output_dir, exist_ok=True)

    results = []

    for shots in args.num_shots_list:
        for test_id, test_code in enumerate(domains):
            domain_label = domain_names[test_code]
            acc_list = []

            for seed in args.seeds:
                set_seed(seed)

                # 加载模型
                model_path = find_model_path(root_results, test_code, seed)
                clip = CLIPWrapper(pretrained_path=args.pretrained_path, device=device)
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
                model.load_state_dict(torch.load(model_path))
                model.eval()

                for cls in class_names:
                    if cls not in model.prompt_learner.context_bank:
                        model.prompt_learner.add_class_prompt(cls)

                # 加载数据
                val_set = dataset[test_id]
                val_set.transform = clip.get_preprocess()
                val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

                if shots > 0:
                    train_sets = []
                    for i in range(len(domains)):
                        if i != test_id:
                            d = dataset[i]
                            d.transform = clip.get_preprocess()
                            few = fewshot_sampler(d, shots)
                            train_sets.append(few)
                    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=args.batch_size, shuffle=True)
                    fine_tune(model, train_loader, device, steps=args.ft_steps, lr=args.ft_lr)

                acc = evaluate_on_domain(model, val_loader, device)
                acc_list.append(acc)
                print(f"[{shots}-shot] Domain={domain_label}, seed={seed} → acc={acc:.2f}")

            results.append({
                "Domain": domain_label,
                "Shots": f"{shots}-shot" if shots > 0 else "Zero-Shot",
                "MeanAcc": np.mean(acc_list),
                "Std": np.std(acc_list)
            })

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "eval_results.csv"), index=False)
    print(f"✅ Results saved to {output_dir}/eval_results.csv")

    # 绘图
    plt.figure(figsize=(10, 5))
    shots = df['Shots'].unique()
    doms = df['Domain'].unique()
    x = np.arange(len(doms))
    bar_width = 0.25

    for i, shot_type in enumerate(shots):
        subset = df[df['Shots'] == shot_type]
        accs = subset.set_index('Domain').loc[doms]['MeanAcc'].values
        plt.bar(x + i * bar_width, accs, width=bar_width, label=shot_type)

    plt.xticks(x + bar_width, doms)
    plt.ylabel("Accuracy (%)")
    plt.title("LODO Evaluation")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eval_bar.png"))
    print(f"📊 Plot saved to {output_dir}/eval_bar.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LODO Evaluation for TAP-CLIP")

    parser.add_argument('--version', type=str, default="version3", help="Experiment version folder name")
    parser.add_argument('--pretrained_path', type=str, default="G:/dsCLIP/open_clip_pytorch_model.bin")
    parser.add_argument('--num_shots_list', type=int, nargs='+', default=[0, 5, 15])
    parser.add_argument('--prompt_len', type=int, default=5)
    parser.add_argument('--attr_lambda', type=float, default=0.05)
    parser.add_argument('--stab_lambda', type=float, default=0.1)
    parser.add_argument('--warmup_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ft_steps', type=int, default=10)
    parser.add_argument('--ft_lr', type=float, default=5e-3)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--prefix_len', type=int, default=5)

    args = parser.parse_args()
    main(args)

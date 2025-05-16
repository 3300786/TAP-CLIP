import os
from torch.multiprocessing import freeze_support
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from models.model_wrapper import FullModel
from models.clip_wrapper import CLIPWrapper
from utils.eval_metrics import evaluate_accuracy
from dataset import get_dataloaders

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def freeze_model_except_prompt(model):
    for name, param in model.named_parameters():
        if "prompt_learner.context_bank" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def fine_tune_on_few_shot(model, loader, device, steps=10, lr=5e-3):
    freeze_model_except_prompt(model)
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for _ in range(steps):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images, labels)
            loss = outputs["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- Config -------- #
    pretrained_path = "G:/dsCLIP/open_clip_pytorch_model.bin"
    epochs = 27
    o_acc = 97.14
    expand = True
    model_path = f"Best Models/best_model_attr_acc{o_acc}.pt"
    seen_class_names = ["Backpack", "Alarm_Clock", "Laptop", "Pen", "Mug"]
    all_class_names = seen_class_names
    prompt_len = 5
    domains = ["Real World", "Art", "Clipart", "Product"]
    num_shots_list = [0, 5, 15]
    batch_size = 32
    ft_steps = 10
    ft_lr = 5e-3

    # -------- Load Model -------- #
    clip_model = CLIPWrapper(pretrained_path=pretrained_path, device=device)


    # -------- Evaluate -------- #
    results = []

    for num_shots in num_shots_list:
        shot_type = f"{num_shots}-shot" if num_shots > 0 else "Zero-Shot"
        for domain in domains:
            print(f"\n🌍 [{shot_type}] Testing on {domain} domain...")
            # ✅ 每轮构建新的 model（防止污染和 hook 丢失）
            model = FullModel(
                class_names=all_class_names,  # 注意这里用 all_class_names！
                clip_wrapper=clip_model,
                prompt_len=prompt_len,
                attr_lambda=1.0,
                stab_lambda=0.1,
                adjustor_method='scale',
                class_specific=True
            ).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            # ✅ 确保 prompt 补全
            for cls in all_class_names:
                if cls not in model.prompt_learner.context_bank:
                    model.prompt_learner.add_class_prompt(cls)
            train_loader, val_loader = get_dataloaders(
                root_dir=f"data/OfficeHomeDataset_10072016/{domain}",
                class_names=all_class_names,
                batch_size=batch_size,
                num_shots=num_shots,
                preprocess=clip_model.get_preprocess()
            )

            # 微调
            if num_shots > 0:
                fine_tune_on_few_shot(model, train_loader, device, steps=ft_steps, lr=ft_lr)

            acc = evaluate_accuracy(model, val_loader, device)
            results.append({"Domain": domain, "Shots": shot_type, "Accuracy": acc})

    # -------- Save -------- #
    df = pd.DataFrame(results)
    from datetime import datetime
    tag = datetime.now().strftime("%Y%m%d_%H%M")
    df.to_csv(f"visible results/cross_domain_results_{tag}.csv", index=False)
    print("✅ Results saved to cross_domain_results.csv")

    # -------- Plot -------- #
    plt.figure(figsize=(10, 5))
    domains = df['Domain'].unique()
    shots = df['Shots'].unique()
    bar_width = 0.25
    x = np.arange(len(domains))

    for i, shot_type in enumerate(shots):
        subset = df[df['Shots'] == shot_type]
        accs = subset.set_index('Domain').loc[domains]['Accuracy'].values
        plt.bar(x + i * bar_width, accs, width=bar_width, label=shot_type)

    plt.xticks(x + bar_width * (len(shots) - 1) / 2, domains)
    plt.title("Cross-Domain Accuracy (Bar Chart)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"visible results/cross_domain_accuracy_bar_{epochs}_{o_acc}_{expand}.png")
    print("✅ Plot saved to cross_domain_accuracy_bar.png")


if __name__ == "__main__":
    freeze_support()
    main()

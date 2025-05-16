import os
from torch.multiprocessing import freeze_support
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from models.model_wrapper import FullModel
from models.clip_wrapper import CLIPWrapper
from utils.eval_metrics import evaluate_accuracy
from dataset import get_dataloaders

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def freeze_model_except_prompt(model):
    for name, param in model.named_parameters():
        param.requires_grad = "prompt_learner.context_bank" in name

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

def generate_output_paths(version: str):
    tag = datetime.now().strftime("%Y%m%d_%H%M")
    base = f"results/{version}_{tag}"
    paths = {
        "base": base,
        "csv": os.path.join(base, "csv"),
        "plot": os.path.join(base, "plots")
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    version = "May15"

    # -------- Config -------- #
    pretrained_path = "G:/dsCLIP/open_clip_pytorch_model.bin"
    model_acc = 97.14
    model_path = f"results/version2_20250516_144859/models/best_model_attr_acc{model_acc}.pt"

    class_names = ["Backpack", "Alarm_Clock", "Laptop", "Pen", "Mug"]
    prompt_len = 5
    domains = ["Real World", "Art", "Clipart", "Product"]
    num_shots_list = [0, 5, 15]
    batch_size = 32
    ft_steps = 10
    ft_lr = 5e-3

    output_paths = generate_output_paths(version)

    # -------- Model & CLIP -------- #
    clip_model = CLIPWrapper(pretrained_path=pretrained_path, device=device)
    results = []

    for num_shots in num_shots_list:
        shot_type = f"{num_shots}-shot" if num_shots > 0 else "Zero-Shot"
        for domain in domains:
            print(f"\n🌍 [{shot_type}] Testing on {domain} domain...")

            # 🔁 每轮新建模型防止状态干扰
            model = FullModel(
                class_names=class_names,
                clip_wrapper=clip_model,
                prompt_len=prompt_len,
                attr_lambda=1.0,
                stab_lambda=0.1,
                adjustor_method='scale',
                class_specific=True
            ).to(device)
            model.load_state_dict(torch.load(model_path))
            model.eval()

            for cls in class_names:
                if cls not in model.prompt_learner.context_bank:
                    model.prompt_learner.add_class_prompt(cls)

            train_loader, val_loader = get_dataloaders(
                root_dir=f"data/OfficeHomeDataset_10072016/{domain}",
                class_names=class_names,
                batch_size=batch_size,
                num_shots=num_shots,
                preprocess=clip_model.get_preprocess()
            )

            if num_shots > 0:
                fine_tune_on_few_shot(model, train_loader, device, steps=ft_steps, lr=ft_lr)

            acc = evaluate_accuracy(model, val_loader, device)
            results.append({"Domain": domain, "Shots": shot_type, "Accuracy": acc})

    # -------- Save CSV -------- #
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_paths["csv"], f"cross_domain_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Results saved to {csv_path}")

    # -------- Bar Chart -------- #
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
    plt.title(f"Cross-Domain Accuracy [{version}]")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_paths["plot"], f"cross_domain_bar_{version}.png")
    plt.savefig(plot_path)
    print(f"✅ Plot saved to {plot_path}")

if __name__ == "__main__":
    freeze_support()
    main()
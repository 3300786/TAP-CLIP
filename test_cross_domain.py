# test_cross_domain.py
import os
from torch.multiprocessing import freeze_support
import torch
import matplotlib.pyplot as plt
import pandas as pd
from models.model_wrapper import FullModel
from models.clip_wrapper import CLIPWrapper
from utils.eval_metrics import evaluate_accuracy
from dataset import get_dataloaders


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- Config -------- #
    pretrained_path = "G:/dsCLIP/open_clip_pytorch_model.bin"
    epochs = 27
    acc = 99.71
    expand = True
    model_path = f"Best Models/best_model_epoch{epochs}_acc{acc}.pt"
    seen_class_names = ["Backpack", "Alarm_Clock", "Laptop", "Pen"]
    all_class_names = seen_class_names + ["Clipboards"]  # Clipboards is unseen
    prompt_len = 5
    domains = ["Real World", "Art", "Clipart", "Product"]
    num_shots_list = [0, 5, 15]
    batch_size = 32

    # -------- Load Model -------- #
    clip_model = CLIPWrapper(pretrained_path=pretrained_path, device=device)
    model = FullModel(
        class_names=seen_class_names,
        clip_wrapper=clip_model,
        prompt_len=prompt_len,
        attr_lambda=1.0,
        stab_lambda=0.1,
        adjustor_method='scale',
        class_specific=True
    ).to(device)
    state_dict = torch.load(model_path)
    converted = {}

    # 将旧模型的 context_emb 拆分到 context_bank
    if "prompt_learner.context_emb" in state_dict:
        old_ctx = state_dict["prompt_learner.context_emb"]  # 可能是 [1, prompt_len, dim] 或 [n_cls, prompt_len, dim]
        n_cls = old_ctx.size(0)
        for i, cls_name in enumerate(seen_class_names):
            converted[f"prompt_learner.context_bank.{cls_name}"] = old_ctx[i]

    # 如果你还想保留 class_token_emb，可以忽略或跳过（因为 token_bank 是自动生成的）

    # 过滤其余匹配参数
    for k, v in state_dict.items():
        if "prompt_learner" not in k:
            converted[k] = v

    # 最终加载
    model.load_state_dict(converted, strict=False)

    model.eval()

    for cls in all_class_names:
        if cls not in model.prompt_learner.context_bank:
            model.prompt_learner.add_class_prompt(cls)

    # -------- Evaluate -------- #
    results = []

    for num_shots in num_shots_list:
        shot_type = f"{num_shots}-shot" if num_shots > 0 else "Zero-Shot"
        for domain in domains:
            print(f"\n🌍 [{shot_type}] Testing on {domain} domain...")
            val_loader = get_dataloaders(
                root_dir=f"data/OfficeHomeDataset_10072016/{domain}",
                class_names=all_class_names,
                batch_size=batch_size,
                num_shots=num_shots,
                preprocess=clip_model.get_preprocess()
            )[1]
            # few shot check
            acc = evaluate_accuracy(model, val_loader, device)
            results.append({"Domain": domain, "Shots": shot_type, "Accuracy": acc})

    # -------- Save -------- #
    df = pd.DataFrame(results)
    df.to_csv(f"visible results/cross_domain_results_{epochs}_{acc}_{expand}.csv", index=False)
    print("✅ Results saved to cross_domain_results.csv")

    # -------- Plot -------- #
    # -------- Bar Chart 绘图 -------- #
    import numpy as np

    plt.figure(figsize=(10, 5))
    domains = df['Domain'].unique()
    shots = df['Shots'].unique()
    bar_width = 0.35
    x = np.arange(len(domains))  # x 轴为 domain 编号

    for i, shot_type in enumerate(shots):
        subset = df[df['Shots'] == shot_type]
        accs = subset.set_index('Domain').loc[domains]['Accuracy'].values
        plt.bar(x + i * bar_width, accs, width=bar_width, label=shot_type)

    plt.xticks(x + bar_width * (len(shots) - 1) / 3, domains)
    plt.title("Cross-Domain Accuracy (Bar Chart)")
    plt.ylabel("Accuracy (%)")
    plt.ylim(80, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"visible results/cross_domain_accuracy_bar_{epochs}_{acc}_{expand}.png")
    print("✅ Plot saved to cross_domain_accuracy_bar.png")


if __name__ == "__main__":
    freeze_support()
    main()

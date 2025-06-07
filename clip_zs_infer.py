import os
import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import open_clip

# ---------- Step 1: Load CLIP ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='', device=device)
checkpoint = torch.load("open_clip_pytorch_model_32.bin", map_location=device)
model.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# ---------- Step 2: Class Names ----------
domain_map = {"A": "Art", "C": "Clipart", "P": "Product", "R": "Real World"}
domains = list(domain_map.values())


def get_classnames(root, domain):
    subfolder = os.path.join(root, domain)
    classes = sorted([d.name for d in os.scandir(subfolder) if d.is_dir()])
    return classes


# ---------- Step 3: Evaluation ----------
def evaluate(domain_root, classnames):
    test_dataset = ImageFolder(domain_root, transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(domain_root)
    prompts = [f"a photo of a {c}" for c in classnames]
    # prompts = [f"{c}" for c in classnames]
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    # print(len(test_loader))
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing on {domain_root}"):
            images = images.to(device)
            labels = labels.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = image_features @ text_features.T
            preds = logits.argmax(dim=-1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)
            # print(total)
    # print(f'--{total}--')
    acc = 100.0 * correct / total
    return acc


# ---------- Step 4: Run LODO ----------

root = f'data/OfficeHomeDataset_10072016'
all_acc = {}
for test_domain in domains:
    classnames = get_classnames(root, test_domain)
    acc = evaluate(os.path.join(root, test_domain), classnames)
    all_acc[test_domain] = acc
    print(f"✅ Zero-shot Acc on {test_domain}: {acc:.2f}%")

# ---------- Final Summary ----------
avg_acc = sum(all_acc.values()) / len(all_acc)
print(len(all_acc))
print(f"\n🔍 Average Zero-shot Accuracy: {avg_acc:.2f}%")

print("\n📊 Zero-shot Accuracy Summary:")
for k, v in all_acc.items():
    print(f"{k:12}: {v:.2f}%")

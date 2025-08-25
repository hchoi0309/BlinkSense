import argparse, json, random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, roc_auc_score

from models.simple_resnet_eye import SimpleResNetEye

# For reproducibility
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Dataset
class MRLDataset(Dataset):
    """
    Expects CSV with (at least): subject_id, img_relpath, eye_state (0/1).
    root_dir should be the prefix so (root_dir / img_relpath) is the full path.
    """
    def __init__(self, df: pd.DataFrame, root_dir: str, eye_size: int = 64, train: bool = True):
        self.df = df.reset_index(drop=True)
        self.root = Path(root_dir)
        self.eye_size = eye_size

        if train:
            # Data augmentation is performed only during the training phase
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomRotation(5),                         # tiny rotation
                transforms.ColorJitter(brightness=0.2, contrast=0.2), # mild lighting jitter for webcam variability
                transforms.ToTensor(),                                # [0,1]
                transforms.Normalize(mean=[0.5], std=[0.5])           # -> [-1,1]
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def __len__(self): return len(self.df) # The number of samples

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.root / row["img_relpath"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        if img.shape != (self.eye_size, self.eye_size):
            img = cv2.resize(img, (self.eye_size, self.eye_size), interpolation=cv2.INTER_AREA)
        x = self.tf(img)                 # [1, H, W]
        y = int(row["eye_state"])        # 0/1
        return x, y


# Subject-exclusive split
def subject_split(df: pd.DataFrame, splits_json: Optional[str], val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    If splits_json exists, load subjects from it.
    Else, create subject-exclusive splits and save a JSON next to the CSV.
    """
    if splits_json and Path(splits_json).exists():
        sj = json.loads(Path(splits_json).read_text())
        train_ids, val_ids, test_ids = sj["train"], sj["val"], sj["test"]
    else:
        subjects = sorted(df["subject_id"].astype(str).unique().tolist())
        rng = random.Random(seed); rng.shuffle(subjects) # Reproducible shuffling of subjects
        n = len(subjects)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int(n * val_ratio))
        test_ids = subjects[:n_test]
        val_ids  = subjects[n_test:n_test+n_val]
        train_ids= subjects[n_test+n_val:]
        # Save for reproducibility if a path is provided
        if splits_json:
            Path(splits_json).parent.mkdir(parents=True, exist_ok=True)
            Path(splits_json).write_text(json.dumps(
                {"train": train_ids, "val": val_ids, "test": test_ids}, indent=2))
    # Split the dataframe
    tr = df[df["subject_id"].astype(str).isin(train_ids)].reset_index(drop=True)
    va = df[df["subject_id"].astype(str).isin(val_ids)].reset_index(drop=True)
    te = df[df["subject_id"].astype(str).isin(test_ids)].reset_index(drop=True)
    return tr, va, te, (train_ids, val_ids, test_ids)


# Train / Evaluate
def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

@torch.no_grad() # Disable gradient tracking for faster inference
def evaluate(model, loader, device, criterion):
    model.eval()
    total = 0.0
    all_y, all_pred = [], []
    all_prob = []
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits, _ = model(x)
        loss = criterion(logits, y)
        total += loss.item() * x.size(0)
        pred = logits.argmax(dim=1).cpu().numpy()
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_prob.append(prob)
        all_pred.append(pred); all_y.append(y.cpu().numpy())
    all_pred = np.concatenate(all_pred); all_y = np.concatenate(all_y)
    all_prob = np.concatenate(all_prob)
    try:
        auroc = roc_auc_score(all_y, all_prob)
    except ValueError:
        auroc = float("nan")
    acc = accuracy_score(all_y, all_pred)
    f1  = f1_score(all_y, all_pred)
    cm  = confusion_matrix(all_y, all_pred)
    rep = classification_report(all_y, all_pred, digits=3)
    return total / len(loader.dataset), acc, f1, cm, rep, auroc

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/mrl/meta/mrl_index.csv", help="MRL index CSV path")
    ap.add_argument("--root", default="data/mrl", help="Dataset root (prefix for img_relpath)")
    ap.add_argument("--splits_json", default="data/mrl/meta/splits.json", help="Where to load/save subject splits")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--eye_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_dir", default="outputs")
    args = ap.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    pin_mem = (device.type == "cuda")

    # Load CSV & basic checks
    df = pd.read_csv(args.csv)
    for col in ("subject_id", "img_relpath", "eye_state"):
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    # Subject-exclusive split (load or create)
    tr_df, va_df, te_df, subj_splits = subject_split(df, args.splits_json, seed=args.seed)
    tr_n = tr_df["subject_id"].nunique(); va_n = va_df["subject_id"].nunique(); te_n = te_df["subject_id"].nunique()
    print(f"Subjects -> train:{tr_n}  val:{va_n}  test:{te_n}")

    # DataLoaders
    ds_tr = MRLDataset(tr_df, args.root, eye_size=args.eye_size, train=True)
    ds_va = MRLDataset(va_df, args.root, eye_size=args.eye_size, train=False)
    ds_te = MRLDataset(te_df, args.root, eye_size=args.eye_size, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                       num_workers=args.num_workers, pin_memory=pin_mem, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=pin_mem)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=pin_mem)

    # Model / loss / optimizer
    model = SimpleResNetEye(emb_dim=64, num_classes=2).to(device)

    # Simple class weights (in case open/closed are imbalanced)
    y_tr = tr_df["eye_state"].astype(int).values
    n_open = y_tr.sum(); n_total = len(y_tr); n_closed = n_total - n_open
    w_closed = 1.0
    w_open = float(n_closed / max(1, n_open))   # >1 if open is minority
    class_weights = torch.tensor([w_closed, w_open], dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train: save best on val accuracy
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "simple_resnet_eye_best.pt"
    best_acc = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, dl_tr, device, optimizer, criterion)
        va_loss, va_acc, va_f1, _, _, va_auroc = evaluate(model, dl_va, device, criterion)

        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
              f"val_acc={va_acc:.3f} | val_f1={va_f1:.3f} | val_auroc={va_auroc:.3f}")
        history.append({"epoch": int(epoch),
                        "train_loss": float(tr_loss),
                        "val_loss": float(va_loss),
                        "val_acc": float(va_acc),
                        "val_f1": float(va_f1),
                        "val_auroc": float(va_auroc)})

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(),
                        "args": vars(args),
                        "subjects": subj_splits,
                        "best_val_acc": float(best_acc),
                        "epoch": int(epoch)}, ckpt_path)
            print(f"\t-> saved best model to {ckpt_path} (val_acc={best_acc:.3f})")

    # Test with the best checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    te_loss, te_acc, te_f1, te_cm, te_rep, te_auroc = evaluate(model, dl_te, device, criterion)
    # Persist metrics to disk
    train_ids, val_ids, test_ids = subj_splits
    metrics = {
        "args": vars(args),
        "subjects": {"train": train_ids, "val": val_ids, "test": test_ids},
        "best_val_acc": float(best_acc),
        "test": {"loss": float(te_loss), "acc": float(te_acc), "f1": float(te_f1), "auroc": float(te_auroc)},
        "confusion_matrix": te_cm.tolist(),
        "history": history
    }
    metrics_path = save_dir / "train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print("\nTest Results:\n")
    print(f"loss={te_loss:.4f} | acc={te_acc:.3f} | f1={te_f1:.3f} | auroc={te_auroc:.3f}")
    print("Confusion matrix:\n", te_cm)
    print("Classification report:\n", te_rep)


if __name__ == "__main__":
    main()
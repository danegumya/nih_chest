import os
import glob
import json
import random
import gc
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torchvision
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.backends.cudnn.benchmark = True

data_dir = '../input/data'
img_size = 256
batch_size = 32
epochs = 12
lr = 1e-4
device = torch.device("cuda")
patience = 3
grad_clip = 1.0
classes = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
    'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
    'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CXRDataset(Dataset):
    def __init__(self, df, y, transform=None):
        self.df = df
        self.y = y
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx]["path"])
        if img.mode not in ("RGB", "L"): img = img.convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.tensor(self.y[idx], dtype=torch.float32)

def main():
    seed_everything()

    all_image_paths = {os.path.basename(x): x for x in glob.glob(os.path.join(data_dir, '**', '*.png'), recursive=True)}

    df = pd.read_csv(glob.glob(os.path.join(data_dir, '**', 'Data_Entry_2017.csv'), recursive=True)[0])
    df['path'] = df['Image Index'].map(all_image_paths)
    df = df.dropna(subset=['path']).reset_index(drop=True)

    df['labels'] = df['Finding Labels'].apply(lambda x: [i.strip().replace("Pleural Thickening", "Pleural_Thickening") for i in x.split('|') if i != 'No Finding'])
    mlb = MultiLabelBinarizer(classes=classes)
    y = mlb.fit_transform(df['labels'])

    train_val_list = set(open(glob.glob(os.path.join(data_dir, '**', 'train_val_list.txt'), recursive=True)[0]).read().splitlines())
    test_list = set(open(glob.glob(os.path.join(data_dir, '**', 'test_list.txt'), recursive=True)[0]).read().splitlines())

    df['is_test'] = df['Image Index'].isin(test_list)
    df_trainval = df[~df['is_test']].reset_index(drop=True)
    y_trainval = y[~df['is_test']]
    df_test = df[df['is_test']].reset_index(drop=True)
    y_test = y[df['is_test']]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    tr_idx, va_idx = next(gss.split(df_trainval, groups=df_trainval['Patient ID']))

    df_train, y_train = df_trainval.iloc[tr_idx], y_trainval[tr_idx]
    df_val, y_val = df_trainval.iloc[va_idx], y_trainval[va_idx]

    pos_weights = torch.tensor((len(y_train) - y_train.sum(axis=0)) / np.maximum(y_train.sum(axis=0), 1), dtype=torch.float32)

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        norm
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        norm
    ])

    train_loader = DataLoader(CXRDataset(df_train, y_train, train_tf), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(CXRDataset(df_val, y_val, val_tf), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = torchvision.models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.classifier.in_features, len(classes)))
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))
    scaler = GradScaler('cuda')

    best_auc = 0
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        avg_loss = 0

        if epoch == 0:
            for p in model.features.parameters(): p.requires_grad = False
        elif epoch == 1:
            for p in model.features.parameters(): p.requires_grad = True

        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                logits = model(imgs)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            avg_loss += loss.item()

        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                with autocast('cuda'):
                    preds.append(torch.sigmoid(model(imgs)).cpu().numpy())
                acts.append(targets.numpy())

        preds = np.vstack(preds)
        acts = np.vstack(acts)

        aucs = []
        for i in range(len(classes)):
            try: aucs.append(roc_auc_score(acts[:, i], preds[:, i]))
            except: pass
        mean_auc = np.mean(aucs)

        print(f"Epoch {epoch+1} | Loss: {avg_loss/len(train_loader):.4f} | Val AUC: {mean_auc:.4f}")

        scheduler.step()

        if mean_auc > best_auc:
            best_auc = mean_auc
            no_improve = 0
            torch.save(model.state_dict(), "densenet121_best.pth")

            thresholds = {}
            for i, cls in enumerate(classes):
                best_t, best_f1 = 0.5, 0
                for t in np.linspace(0.1, 0.9, 50):
                    f1 = f1_score(acts[:, i], (preds[:, i] >= t).astype(int))
                    if f1 > best_f1: best_f1, best_t = f1, t
                thresholds[cls] = best_t

            with open("thresholds.json", "w") as f: json.dump(thresholds, f)
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break

    if df_test is not None:
        model.load_state_dict(torch.load("densenet121_best.pth"))
        model.eval()
        test_loader = DataLoader(CXRDataset(df_test, y_test, val_tf), batch_size=batch_size, shuffle=False, num_workers=4)

        preds = []
        with torch.no_grad():
            for imgs, _ in test_loader:
                with autocast('cuda'):
                    preds.append(torch.sigmoid(model(imgs.to(device))).cpu().numpy())

        preds = np.vstack(preds)
        with open("thresholds.json", "r") as f: th = json.load(f)

        sub = df_test[['Image Index']].copy()
        for i, cls in enumerate(classes):
            sub[cls] = (preds[:, i] >= th[cls]).astype(int)
        sub.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    main()
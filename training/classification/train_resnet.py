"""
"""

import os
import json
import logging
import argparse
import warnings
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ============================================================
#
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train microstructure classifier")

    # # modelArgs
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "vgg16", "vgg19"],
                        help="Model architecture")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Use ImageNet pretrained weights")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes")

    #
    parser.add_argument("--data", type=str, required=True,
                        help="Path to dataset directory (organized by class subfolders)")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")

    # # trainArgs
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0001,
                        help="Weight decay")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing")

    #
    parser.add_argument("--kfold", type=int, default=10,
                        help="K-fold cross-validation (1 = single train/val split)")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Validation ratio (if kfold=1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    #
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device (cuda:0, cpu)")
    parser.add_argument("--output", type=str, default="results/classification",
                        help="Output directory")

    return parser.parse_args()


# ============================================================
# modelbuild
# ============================================================

def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """
    buildclassifymodel。

    Args:
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name in ("vgg16", "vgg19"):
        vgg_fn = models.vgg16 if model_name == "vgg16" else models.vgg19
        if pretrained:
            model = vgg_fn(pretrained=True)
        else:
            model = vgg_fn(pretrained=False)
        #
        classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        model.classifier = classifier
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


# ============================================================
#
# ============================================================

def get_transforms(img_size: int, is_train: bool = True):
    """
    """
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


def load_dataset(data_dir: str, img_size: int) -> Tuple[datasets.ImageFolder, List, List]:
    """
    loadimageclassifydataset。

    Returns:
        (full_dataset, targets_list, class_names)
    """
    #
    temp_transform = get_transforms(img_size, is_train=False)
    dataset = datasets.ImageFolder(data_dir, transform=temp_transform)

    class_names = dataset.classes
    targets = [s[1] for s in dataset.samples]
    logger.info(f"Dataset loaded: {len(dataset)} images, {len(class_names)} classes")
    for i, name in enumerate(class_names):
        count = targets.count(i)
        logger.info(f"  Class {i} ({name}): {count} images")

    return dataset, targets, class_names


# ============================================================
#
# ============================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Module functionality."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return {"loss": epoch_loss, "accuracy": epoch_acc}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """validate。"""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    val_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "loss": val_loss, "accuracy": val_acc,
        "precision": val_precision, "recall": val_recall, "f1": val_f1,
    }


# ============================================================
#
# ============================================================

def run_fold(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device: torch.device,
    fold: int,
    class_names: List[str],
    output_dir: str,
) -> Dict[str, Any]:
    """
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    best_val_f1 = 0.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)

        record = {"epoch": epoch + 1, **train_metrics, **{
            f"val_{k}": v for k, v in val_metrics.items()
        }}
        history.append(record)

        logger.info(
            f"  Fold {fold} Epoch {epoch+1}/{args.epochs}: "
            f"train_loss={train_metrics['loss']:.4f}, train_acc={train_metrics['accuracy']:.4f}, "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        #
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"  Fold {fold} early stopping at epoch {epoch+1}")
                break

    #
    if best_state is not None:
        model.load_state_dict(best_state)

    #
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    # # save fold result
    fold_dir = os.path.join(output_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    #
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(fold_dir, f"confusion_matrix_fold_{fold}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # # savemodelweights
    torch.save(model.state_dict(),
               os.path.join(fold_dir, f"{args.model}_fold_{fold}_best.pth"))

    #
    pd.DataFrame(history).to_csv(
        os.path.join(fold_dir, f"training_metrics_fold_{fold}.csv"), index=False
    )

    return {
        "fold": fold,
        "best_val_f1": float(best_val_f1),
        "val_metrics": validate(model, val_loader, criterion, device),
        "confusion_matrix": cm.tolist(),
    }


def train(args):
    """Module functionality."""
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    #
    dataset, targets, class_names = load_dataset(args.data, args.img_size)
    num_classes = len(class_names)

    #
    with open(os.path.join(args.output, "class_indices.json"), "w") as f:
        json.dump({name: i for i, name in enumerate(class_names)}, f, indent=2)

    #
    if args.kfold > 1:
        kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True,
                                random_state=args.seed)
        splits = list(kfold.split(np.arange(len(dataset)), targets))
    else:
        #
        indices = np.arange(len(dataset))
        np.random.seed(args.seed)
        np.random.shuffle(indices)
        val_size = int(len(dataset) * args.val_ratio)
        splits = [(indices[val_size:], indices[:val_size])]

    logger.info(f"Starting {len(splits)}-fold cross-validation")

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold}/{len(splits)}")
        logger.info(f"{'='*60}")

        #
        model = build_model(args.model, num_classes, args.pretrained).to(device)

        #
        train_dataset = datasets.ImageFolder(
            args.data, transform=get_transforms(args.img_size, is_train=True)
        )
        train_subset = Subset(train_dataset, train_idx)
        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size,
            shuffle=True, num_workers=4, pin_memory=True,
        )

        val_dataset = datasets.ImageFolder(
            args.data, transform=get_transforms(args.img_size, is_train=False)
        )
        val_subset = Subset(val_dataset, val_idx)
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size,
            shuffle=False, num_workers=4, pin_memory=True,
        )

        result = run_fold(model, train_loader, val_loader, args, device,
                         fold, class_names, args.output)
        fold_results.append(result)

    #
    f1_scores = [r["best_val_f1"] for r in fold_results]
    acc_scores = [r["val_metrics"]["accuracy"] for r in fold_results]

    summary = {
        "model": args.model,
        "pretrained": args.pretrained,
        "num_classes": num_classes,
        "kfold": len(fold_results),
        "mean_accuracy": float(np.mean(acc_scores)),
        "std_accuracy": float(np.std(acc_scores)),
        "mean_f1": float(np.mean(f1_scores)),
        "std_f1": float(np.std(f1_scores)),
        "fold_results": fold_results,
    }

    with open(os.path.join(args.output, f"{args.model}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Training Complete")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Mean Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    logger.info(f"  Mean F1: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    logger.info(f"  Results saved to: {args.output}")

    #
    avg_cm = np.mean([np.array(r["confusion_matrix"]) for r in fold_results], axis=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_cm, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Average Confusion Matrix - {args.model}")
    plt.savefig(os.path.join(args.output, f"avg_confusion_matrix_{args.model}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    #
    plt.figure(figsize=(10, 6))
    for r in fold_results:
        history = pd.read_csv(
            os.path.join(args.output, f"fold_{r['fold']}",
                         f"training_metrics_fold_{r['fold']}.csv")
        )
        plt.plot(history["epoch"], history["val_accuracy"],
                 label=f"Fold {r['fold']}", alpha=0.7)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title(f"Fold Accuracy Curves - {args.model}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, f"fold_accuracy_curves_{args.model}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("training_classification.log"),
            logging.StreamHandler(),
        ],
    )
    args = parse_args()
    train(args)

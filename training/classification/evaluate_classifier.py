"""
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate microstructure classifier")

    #
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to model weights (.pth)")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "vgg16", "vgg19"],
                        help="Model architecture (used if --weights provided)")

    #
    parser.add_argument("--compare", type=str, default=None,
                        help="Comma-separated model names for comparison (e.g., resnet18,resnet50)")

    #
    parser.add_argument("--data", type=str, required=True,
                        help="Path to test dataset directory")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image size")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")

    #
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device")
    parser.add_argument("--output", type=str, default="results/evaluation",
                        help="Output directory")

    return parser.parse_args()


def get_transform(img_size: int):
    """Module functionality."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_model(model_name: str, num_classes: int, weights_path: str, device: torch.device):
    """Module functionality."""
    if model_name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name in ("vgg16", "vgg19"):
        vgg_fn = models.vgg16 if model_name == "vgg16" else models.vgg19
        model = vgg_fn(pretrained=False)
        classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
        model.classifier = classifier
    else:
        raise ValueError(f"Unknown model: {model_name}")

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    """
    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision_macro": float(precision_score(all_labels, all_preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(all_labels, all_preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "n_samples": len(all_labels),
    }

    #
    if len(class_names) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(all_labels, all_probs[:, 1]))
        except:
            metrics["roc_auc"] = None
    else:
        try:
            metrics["roc_auc"] = float(roc_auc_score(all_labels, all_probs, multi_class="ovr"))
        except:
            metrics["roc_auc"] = None

    #
    per_class = {}
    for i, name in enumerate(class_names):
        y_true = (all_labels == i).astype(int)
        y_pred = (all_preds == i).astype(int)
        per_class[name] = {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "support": int(y_true.sum()),
        }
    metrics["per_class"] = per_class

    return metrics


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                          title: str, save_path: str):
    """Module functionality."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(results: Dict[str, Dict], class_names: List[str],
                    save_path: str):
    """Module functionality."""
    model_names = list(results.keys())
    metrics_list = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    x = np.arange(len(metrics_list))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(model_names):
        values = [results[name].get(m, 0) for m in metrics_list]
        ax.bar(x + i * width, values, width, label=name)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate(args):
    """Module functionality."""
    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    #
    test_dataset = datasets.ImageFolder(args.data, transform=get_transform(args.img_size))
    class_names = test_dataset.classes
    num_classes = len(class_names)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    logger.info(f"Test set: {len(test_dataset)} images, {num_classes} classes")

    #
    if args.compare:
        model_names = [m.strip() for m in args.compare.split(",")]
        all_results = {}

        for model_name in model_names:
            weights_dir = os.path.join(args.output, "..", "classification", model_name)
            weights_path = None
            #
            for f in os.listdir(weights_dir) if os.path.isdir(weights_dir) else []:
                if f.endswith("_best.pth"):
                    weights_path = os.path.join(weights_dir, f)
                    break

            if weights_path is None:
                logger.warning(f"No weights found for {model_name}, skipping")
                continue

            logger.info(f"Evaluating {model_name}...")
            model = load_model(model_name, num_classes, weights_path, device)
            metrics = evaluate_model(model, test_loader, device, class_names)
            all_results[model_name] = metrics

            cm = np.array(metrics["confusion_matrix"])
            plot_confusion_matrix(
                cm, class_names, f"Confusion Matrix - {model_name}",
                os.path.join(args.output, f"cm_{model_name}.png"),
            )

        #
        plot_comparison(
            all_results, class_names,
            os.path.join(args.output, "model_comparison.png"),
        )

        #
        with open(os.path.join(args.output, "comparison_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        #
        print(f"\n{'Model':<15} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
        print("-" * 55)
        for name, m in all_results.items():
            print(f"{name:<15} {m['accuracy']:>8.4f} {m['precision_macro']:>8.4f} "
                  f"{m['recall_macro']:>8.4f} {m['f1_macro']:>8.4f}")

    #
    else:
        model = load_model(args.model, num_classes, args.weights, device)
        metrics = evaluate_model(model, test_loader, device, class_names)

        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(
            cm, class_names, f"Confusion Matrix - {args.model}",
            os.path.join(args.output, f"cm_{args.model}.png"),
        )

        with open(os.path.join(args.output, "evaluation_results.json"), "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"\nModel: {args.model}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall:    {metrics['recall_macro']:.4f}")
        print(f"  F1:        {metrics['f1_macro']:.4f}")
        if metrics.get("roc_auc"):
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    evaluate(args)

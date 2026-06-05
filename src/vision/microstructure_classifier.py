"""
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

#
_torch = None
_models = None
_transforms = None
_nn = None


def _lazy_import():
    global _torch, _models, _transforms, _nn
    if _torch is None:
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
            _torch = torch
            _nn = nn
            _models = models
            _transforms = transforms
        except ImportError as e:
            raise ImportError(
                f"Microstructure classifier requires PyTorch and torchvision: {e}\n"
                "Please run: pip install torch torchvision"
            )


#
DEFAULT_CLASSES = [
    "equiaxed",
    "lamellar",
    "bimodal",
    "widmanstatten",
    "acicular",
    "cellular",
    "basketweave",
    "martensitic",
]

# ImageNet standardizeArgs
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(arch: str, num_classes: int, weights_path: Optional[str] = None):
    """
    buildclassifymodel。

    Args:
    """
    _lazy_import()

    if arch == "resnet18":
        model = _models.resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = _nn.Linear(in_features, num_classes)
    elif arch == "resnet50":
        model = _models.resnet50(pretrained=False)
        in_features = model.fc.in_features
        model.fc = _nn.Linear(in_features, num_classes)
    elif arch in ("vgg16", "vgg19"):
        vgg_fn = _models.vgg16 if arch == "vgg16" else _models.vgg19
        model = vgg_fn(pretrained=False)
        classifier = _nn.Sequential(
            _nn.Linear(512 * 7 * 7, 4096),
            _nn.ReLU(True),
            _nn.Dropout(0.5),
            _nn.Linear(4096, 4096),
            _nn.ReLU(True),
            _nn.Dropout(0.5),
            _nn.Linear(4096, num_classes),
        )
        model.classifier = classifier
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    if weights_path and os.path.exists(weights_path):
        state_dict = _torch.load(weights_path, map_location="cpu",
                                 weights_only=True)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded weights from {weights_path}")

    return model


def get_transform(img_size: int = 224):
    """Module functionality."""
    _lazy_import()
    return _transforms.Compose([
        _transforms.Resize(int(img_size * 1.14)),
        _transforms.CenterCrop(img_size),
        _transforms.ToTensor(),
        _transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class MicrostructureClassifier:
    """
    """

    def __init__(
        self,
        model_path: str,
        arch: str = "resnet50",
        class_names: Optional[List[str]] = None,
        img_size: int = 224,
        device: str = "auto",
    ):
        _lazy_import()

        self.arch = arch
        self.img_size = img_size
        self.class_names = class_names or DEFAULT_CLASSES
        self.num_classes = len(self.class_names)

        #
        if device == "auto":
            self.device = _torch.device(
                "cuda:0" if _torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = _torch.device(device)

        # # buildmodel
        self.model = build_model(arch, self.num_classes, model_path)
        self.model.to(self.device)
        self.model.eval()

        #
        self.transform = get_transform(img_size)

        logger.info(
            f"[MicrostructureClassifier] arch={arch}, "
            f"classes={self.num_classes}, device={self.device}"
        )

    def classify(
        self,
        image: str or np.ndarray or Image.Image,
        top_k: int = 3,
    ) -> List[Tuple[str, float]]:
        """
        """
        #
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        #
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        # # inference
        with _torch.no_grad():
            outputs = self.model(tensor)
            probs = _torch.softmax(outputs, dim=1)

        # # Top-K result
        topk_probs, topk_indices = _torch.topk(probs, min(top_k, self.num_classes))

        results = []
        for i in range(len(topk_indices[0])):
            idx = topk_indices[0][i].item()
            conf = topk_probs[0][i].item()
            label = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
            results.append((label, round(conf, 4)))

        return results

    def classify_batch(
        self,
        images: List[str or np.ndarray or Image.Image],
    ) -> List[Dict[str, Any]]:
        """
        """
        results = []
        for img in images:
            top_k = self.classify(img, top_k=3)
            results.append({
                "prediction": top_k[0][0],
                "confidence": top_k[0][1],
                "top_k": top_k,
            })
        return results

    def get_class_distribution(
        self,
        images: List[str or np.ndarray or Image.Image],
    ) -> Dict[str, int]:
        """
        """
        distribution = {name: 0 for name in self.class_names}
        for img in images:
            pred = self.classify(img, top_k=1)[0][0]
            distribution[pred] = distribution.get(pred, 0) + 1
        return distribution

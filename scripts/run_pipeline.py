"""
LitExtract End-to-End Pipeline — 6-Stage Fully Local Execution

  Stage 1  : YOLO Detection          (VisionAgent, detect_only)
  Stage 2  : Microstructure Classify  (MicrostructureClassifier, ResNet/VGG)
  Stage 3  : Figure-Caption Matching  (FigureMatcher, spatial + OCR)
  Stage 4  : Semantic Extraction      (CoordinatorAgent / LoRAAgent)
  Stage 5  : Knowledge Graph Build    (CoordinatorAgent → KnowledgeAgent)
  Stage 6  : Causal DAG Build         (CoordinatorAgent → KnowledgeAgent CHG)

Stages 4-6 are delegated to CoordinatorAgent which orchestrates
SemanticAgent + KnowledgeAgent internally via local models
(Mistral-7B / Qwen2.5-7B / LLaMA-3-8B via LoRA).

Usage:
    python run_pipeline.py paper.pdf
    python run_pipeline.py paper.pdf --use-local --llm-type qwen \
        --base-model-path models/Qwen2.5-7B-Instruct \
        --lora-path training/llm/checkpoints/lora_qwen
    python run_pipeline.py paper.pdf --cls-model models/cls_resnet50.pth --cls-arch resnet50
    python run_pipeline.py --batch /path/to/pdf_dir/ --output batch_results/
    python run_pipeline.py paper.pdf --skip-classification --skip-match

Output (per PDF):
    - vision_output.json          YOLO detection pages
    - classification_output.json  Microstructure class predictions
    - match_output.json           Figure-caption pairings
    - semantic_output.json        Extracted triples (composition/process/property)
    - knowledge_graph.json        KG nodes & edges
    - causal_graph.json           CHG (causal hypothesis graph)
    - {pdf_name}_kg.html          Interactive KG visualization (pyvis)
    - {pdf_name}_chg.html         Interactive CHG visualization (pyvis)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# add project source, training, and parent dir (for utils.utils) to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARENT_ROOT = PROJECT_ROOT.parent  # D:/python/LitExtract_Agent/
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))
sys.path.insert(0, str(PARENT_ROOT))  # for utils.utils used by pdf_to_images.py

from pdf_parser.pdf_to_images import VisionAgent
from vision.microstructure_classifier import MicrostructureClassifier
from vision.figure_matcher import FigureMatcher
from semantic_extraction.coordinator_agent import CoordinatorAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def _save_json(data: Any, path: Path) -> None:
    """Save data as JSON with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("  Saved %s", path)


# ────────────────────────────────────────────────────────
# Main Pipeline
# ────────────────────────────────────────────────────────

def run_pipeline(
    pdf_path: str,
    output_dir: str,
    *,
    use_local_model: bool = False,
    llm_type: str = "mistral",
    cls_model: str = "models/classification/best.pth",
    cls_arch: str = "resnet50",
    skip_classification: bool = False,
    skip_match: bool = False,
    base_model_path: str = "",
    lora_path: str = "",
) -> Dict[str, Any]:
    """Run the full 6-stage pipeline on a single PDF.

    Parameters
    ----------
    pdf_path : str
        Path to the input PDF file.
    output_dir : str
        Directory for output JSON and HTML files.
    use_local_model : bool
        If True, use a local LoRA-finetuned model instead of API.
    llm_type : str
        LLM family for local inference: mistral / qwen / llama.
    cls_model : str
        Path to microstructure classifier weights (.pth).
    cls_arch : str
        Classifier architecture: resnet18 / resnet50 / vgg16 / vgg19.
    skip_classification : bool
        Skip Stage 2 (microstructure classification).
    skip_match : bool
        Skip Stage 3 (figure-caption matching).
    base_model_path : str
        Path to the base LLM (e.g. models/Mistral-7B-Instruct-v0.3).
        Required when use_local_model=True.
    lora_path : str
        Path to the LoRA adapter checkpoint.
        Required when use_local_model=True.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Vision (YOLO detection, detect_only) ──
    logger.info("--- Stage 1: YOLO Detection ---")
    vision = VisionAgent(detect_only=True)
    pages: List[Dict[str, Any]] = vision.run(pdf_path)
    _save_json(pages, output_path / "vision_output.json")

    # ── Stage 2: Microstructure Classification (optional) ──
    if not skip_classification:
        logger.info("--- Stage 2: Microstructure Classification (arch=%s) ---", cls_arch)
        classifier = MicrostructureClassifier(model_path=cls_model, arch=cls_arch)
        for page in pages:
            for img in page.get("images", []):
                img_path = img.get("image_path")
                if img_path:
                    img["microstructure_class"] = classifier.classify(img_path, top_k=3)

        # Save classification summary
        cls_summary = []
        for page in pages:
            for img in page.get("images", []):
                if "microstructure_class" in img:
                    cls_summary.append({
                        "page": page.get("page_num"),
                        "image": img.get("image_name", ""),
                        "classification": img["microstructure_class"],
                    })
        if cls_summary:
            _save_json(cls_summary, output_path / "classification_output.json")

    # ── Stage 3: Figure-Caption Matching (optional) ──
    if not skip_match:
        logger.info("--- Stage 3: Figure-Caption Matching ---")
        matcher = FigureMatcher()
        pages = matcher.match(pages)

        # Save matching summary
        match_summary = []
        for page in pages:
            for img in page.get("images", []):
                if "matched_caption" in img or "subfigures" in img:
                    match_summary.append({
                        "page": page.get("page_num"),
                        "image": img.get("image_name", ""),
                        "caption": img.get("matched_caption", ""),
                        "subfigures": img.get("subfigures", []),
                    })
        if match_summary:
            _save_json(match_summary, output_path / "match_output.json")

    # ── Stage 4-6: Semantic + KG + DAG via CoordinatorAgent ──
    # CoordinatorAgent orchestrates SemanticAgent + KnowledgeAgent internally.
    # It handles both API (DeepSeek) and local models (Mistral/Qwen/LLaMA + LoRA).

    local_config: Optional[Dict[str, Any]] = None
    if use_local_model:
        if not base_model_path or not lora_path:
            raise ValueError(
                "use_local_model=True requires --base-model-path and --lora-path"
            )
        local_config = {
            "model_type": llm_type,
            "base_model_path": base_model_path,
            "lora_path": lora_path,
        }

    coordinator = CoordinatorAgent(
        use_local_model=use_local_model,
        local_model_config=local_config,
    )

    # Stage 4: Semantic extraction
    logger.info("--- Stage 4: Semantic Extraction (local model) ---")
    semantic = coordinator.run_semantic(pdf_path, pages)
    _save_json(semantic, output_path / "semantic_output.json")

    # Stage 5 + 6: KG + CHG construction
    logger.info("--- Stage 5 & 6: KG + Causal DAG Construction ---")
    kg_result = coordinator.run_knowledge_graph(
        semantic, pdf_path, str(output_path)
    )

    # Save individual KG / CHG outputs
    if kg_result.get("knowledge_graph"):
        _save_json(kg_result["knowledge_graph"],
                   output_path / "knowledge_graph.json")
    if kg_result.get("causal_graph"):
        _save_json(kg_result["causal_graph"],
                   output_path / "causal_graph.json")

    logger.info("Pipeline completed. Results saved to %s", output_dir)

    return {
        "vision": pages,
        "semantic": semantic,
        "kg": kg_result,
    }


def run_batch(
    pdf_dir: str,
    output_dir: str,
    **kwargs,
) -> None:
    """Run the pipeline on a directory of PDFs."""
    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    logger.info("Found %d PDFs in %s", len(pdf_files), pdf_dir)

    for pdf_file in pdf_files:
        pdf_name = pdf_file.stem
        pdf_output = os.path.join(output_dir, pdf_name)
        logger.info("\n>>> Processing %s ...", pdf_file.name)
        try:
            run_pipeline(str(pdf_file), pdf_output, **kwargs)
        except Exception as exc:
            logger.error("Failed to process %s: %s", pdf_file.name, exc)


# ────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LitExtract End-to-End Pipeline (6 stages)"
    )

    # PDF input
    parser.add_argument("pdf", nargs="?", help="Path to a single PDF file")
    parser.add_argument("--batch", help="Path to a directory of PDF files")
    parser.add_argument("--output", default="output", help="Output directory")

    # LLM configuration
    parser.add_argument("--use-local", action="store_true",
                        help="Use local LoRA-finetuned model (bypasses API)")
    parser.add_argument("--llm-type", default="mistral",
                        choices=["mistral", "qwen", "llama"],
                        help="LLM family for local inference")
    parser.add_argument("--base-model-path", default="",
                        help="Path to base LLM (required with --use-local)")
    parser.add_argument("--lora-path", default="",
                        help="Path to LoRA adapter (required with --use-local)")

    # Classification / Matching
    parser.add_argument("--cls-model", default="models/classification/best.pth",
                        help="Path to microstructure classifier weights")
    parser.add_argument("--cls-arch", default="resnet50",
                        choices=["resnet18", "resnet50", "vgg16", "vgg19"],
                        help="Classifier architecture")
    parser.add_argument("--skip-classification", action="store_true",
                        help="Skip Stage 2 (microstructure classification)")
    parser.add_argument("--skip-match", action="store_true",
                        help="Skip Stage 3 (figure-caption matching)")

    args = parser.parse_args()

    kwargs = {
        "use_local_model": args.use_local,
        "llm_type": args.llm_type,
        "cls_model": args.cls_model,
        "cls_arch": args.cls_arch,
        "skip_classification": args.skip_classification,
        "skip_match": args.skip_match,
        "base_model_path": args.base_model_path,
        "lora_path": args.lora_path,
    }

    if args.batch:
        run_batch(args.batch, args.output, **kwargs)
    elif args.pdf:
        run_pipeline(args.pdf, args.output, **kwargs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

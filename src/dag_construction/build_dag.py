"""
DAG Build Pipeline: PDF → Local Semantic Extraction → Co-occurrence → DAG
All extraction runs through local model (Qwen2.5-7B-Instruct + LoRA) via subprocess.
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path

#
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = str(_THIS_FILE.parent.parent.parent)  # LitExtract_Agent/
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "KG_and_Causal", "core"))

#
BLIP_PYTHON = os.environ.get("BLIP_PYTHON", sys.executable)
LOCAL_EXTRACT_SCRIPT = os.path.join(PROJECT_ROOT, "KG_and_Causal", "extract_with_local_model.py")

from .cooccurrence_miner import run_step1, load_semantic_data
from .dag_causal_builder import CausalHypothesisGraph, NODE_ORDER


# ============================================
# Step0: Semantic Extraction (Local Model)
# ============================================

def extract_semantic_from_pdfs(
    pdf_dir: str,
    output_json: str,
    max_papers: int = 9999,
    max_chunks_per_paper: int = 6,
    no_resume: bool = False,
):
    """Run semantic extraction via local model subprocess (BLIP Python env)."""
    cmd = [
        BLIP_PYTHON, LOCAL_EXTRACT_SCRIPT,
        "--pdf_dir", pdf_dir,
        "--output", output_json,
        "--max_papers", str(max_papers),
        "--max_chunks", str(max_chunks_per_paper),
    ]
    if no_resume:
        cmd.append("--no_resume")

    print(f"[Step0] Calling local model subprocess: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=os.path.dirname(LOCAL_EXTRACT_SCRIPT))

    # Load results for summary
    with open(output_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"\n[Step0] Complete! Semantic data saved to: {output_json}")
    print(f"        {len(results)} papers total")
    return results


# ============================================
# Step1+2+3: Build DAG
# ============================================

def build_dag_from_pdf_dir(
    pdf_dir: str,
    output_dir: str,
    min_freq: int = 2,
    apply_constraints: bool = True,
    enforce_dag: bool = True,
    max_papers: int = 9999,
    max_chunks_per_paper: int = 6,
    no_resume: bool = False,
):
    """Full pipeline: PDF dir → semantic extraction → co-occurrence → DAG."""
    os.makedirs(output_dir, exist_ok=True)

    # Step0: Semantic extraction
    semantic_json = os.path.join(output_dir, "semantic_data_local.json")
    if not os.path.exists(semantic_json):
        print("=" * 60)
        print("[Pipeline] Step0: Local model semantic extraction")
        print("=" * 60)
        papers = extract_semantic_from_pdfs(
            pdf_dir, semantic_json,
            max_papers=max_papers,
            max_chunks_per_paper=max_chunks_per_paper,
            no_resume=no_resume,
        )
    else:
        print(f"[Pipeline] Step0 skip (using existing {semantic_json})")
        papers = load_semantic_data(semantic_json)

    print(f"\n[Pipeline] Loaded: {len(papers)} papers")

    # Step1: Co-occurrence → candidate edges
    print("\n" + "=" * 60)
    print("[Pipeline] Step1: Co-occurrence mining → candidate edges")
    print("=" * 60)
    candidates_json = os.path.join(output_dir, "candidate_edges_local.json")
    candidates = run_step1(semantic_json, candidates_json, min_freq=min_freq)

    # Step2+3: Build DAG
    print("\n" + "=" * 60)
    print("[Pipeline] Step2+3: Build causal DAG")
    print("=" * 60)
    chg = CausalHypothesisGraph()
    dag_stats = chg.build_dag_from_candidates(
        candidates,
        apply_constraints=apply_constraints,
        enforce_dag=enforce_dag,
        verbose=True,
    )

    # Save results
    dag_json_path = os.path.join(output_dir, "dag_result_local.json")
    dag_graphml_path = os.path.join(output_dir, "dag_result_local.graphml")
    stats_path = os.path.join(output_dir, "dag_stats_local.json")

    with open(dag_json_path, "w", encoding="utf-8") as f:
        json.dump(chg.export_json(), f, ensure_ascii=False, indent=2)
    chg.export_graphml(dag_graphml_path)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "dag_stats": dag_stats,
            "node_order": NODE_ORDER,
            "is_dag": dag_stats.get("is_dag", False),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("[Pipeline] Complete! Output files:")
    print(f"  - Semantic:     {semantic_json}")
    print(f"  - Candidates:   {candidates_json}")
    print(f"  - DAG JSON:     {dag_json_path}")
    print(f"  - DAG GraphML:  {dag_graphml_path}")
    print(f"  - Stats:        {stats_path}")
    print(f"\nDAG is true DAG: {dag_stats.get('is_dag', False)}")
    print(f"Final size: {dag_stats.get('final_nodes', 0)} nodes, {dag_stats.get('final_edges', 0)} edges")

    return dag_stats


# ============================================
# CLI
# ============================================

def main():
    parser = argparse.ArgumentParser(description="DAG build pipeline: PDF → semantic extraction → DAG")
    parser.add_argument("--pdf_dir", default=os.path.join(PROJECT_ROOT, "Ti-test-segmantic"),
                        help="PDF directory path")
    parser.add_argument("--output_dir", default=os.path.join(PROJECT_ROOT, "KG_and_Causal", "output"),
                        help="Output directory")
    parser.add_argument("--min_freq", type=int, default=2,
                        help="Step1 minimum co-occurrence frequency (default 2)")
    parser.add_argument("--skip_constraints", action="store_true",
                        help="Skip Step2 domain constraint filtering")
    parser.add_argument("--skip_dag", action="store_true",
                        help="Skip Step3 cycle removal (keep cyclic graph)")
    parser.add_argument("--max_papers", type=int, default=9999,
                        help="Max papers to process (default all)")
    parser.add_argument("--max_chunks_per_paper", type=int, default=6,
                        help="Max chunks per paper for local model (default 6)")
    parser.add_argument("--no_resume", action="store_true",
                        help="Don't resume from checkpoint, start fresh")
    parser.add_argument("--semantic_only", action="store_true",
                        help="Only do Step0 semantic extraction, skip DAG")
    parser.add_argument("--dag_only", action="store_true",
                        help="Skip Step0, use existing semantic_data_local.json")

    args = parser.parse_args()

    apply_constraints = not args.skip_constraints
    enforce_dag = not args.skip_dag

    if args.semantic_only:
        output_path = os.path.join(args.output_dir, "semantic_data_local.json")
        extract_semantic_from_pdfs(
            pdf_dir=args.pdf_dir,
            output_json=output_path,
            max_papers=args.max_papers,
            max_chunks_per_paper=args.max_chunks_per_paper,
            no_resume=args.no_resume,
        )
    elif args.dag_only:
        semantic_json = os.path.join(args.output_dir, "semantic_data_local.json")
        if not os.path.exists(semantic_json):
            print(f"Error: semantic data file not found: {semantic_json}")
            print("Run without --dag_only to generate it first.")
            return

        print("[DAG-only] Loading existing semantic data...")
        candidates_json = os.path.join(args.output_dir, "candidate_edges_local.json")
        candidates = run_step1(semantic_json, candidates_json, min_freq=args.min_freq)
        chg = CausalHypothesisGraph()
        stats = chg.build_dag_from_candidates(
            candidates,
            apply_constraints=apply_constraints,
            enforce_dag=enforce_dag,
            verbose=True,
        )
        print(f"\nDAG build complete: {stats}")
    else:
        build_dag_from_pdf_dir(
            pdf_dir=args.pdf_dir,
            output_dir=args.output_dir,
            min_freq=args.min_freq,
            apply_constraints=apply_constraints,
            enforce_dag=enforce_dag,
            max_papers=args.max_papers,
            max_chunks_per_paper=args.max_chunks_per_paper,
            no_resume=args.no_resume,
        )


if __name__ == "__main__":
    main()

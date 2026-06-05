"""
LitExtract Agents — core agent modules for the 6-stage pipeline.

Copied from the parent project (D:/python/LitExtract_Agent/agents/)
and adapted for standalone packaging.

Modules:
    kg_schema        — Node, Edge, Evidence, KnowledgeGraph dataclasses
    vocab            — Controlled vocabulary for property/process/phase/morphology mapping
    kg_builder       — Incremental knowledge graph construction
    causal_builder   — Causal Hypothesis Graph (CHG) construction
    knowledge_agent  — High-level KG + CHG builder from semantic extraction results
    semantic_agent   — Semantic extraction agent (API + local model dual backend)
    local_llm_agent  — Local LLM inference for Qwen2.5-7B / Mistral-7B + LoRA
    kg_visualizer    — Pyvis-based interactive KG and CHG HTML visualization
"""

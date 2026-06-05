# agents/local_llm_agent.py
"""
Local fine-tuned model inference module.
Supports Qwen2.5-7B-Instruct + LoRA for local loading and inference.
Replaces the former SemanticAgent DeepSeek API calls.

Usage:
    agent = LocalLLMAgent(
        base_model_path="D:/models/Qwen2.5-7B-Instruct",
        lora_path="D:/python/Litextract_segmantic_data/lora_out/lora_out_qwen/checkpoint-1632",
        use_4bit=True
    )
    result = agent.extract(pages)
"""

import json
import logging
import re
import os
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Lazy import torch / transformers to avoid errors in environments without them
# ──────────────────────────────────────────────────────────────────────────────
_torch = None
_AutoModelForCausalLM = None
_AutoTokenizer = None
_BitsAndBytesConfig = None
_PeftModel = None


def _lazy_import():
    global _torch, _AutoModelForCausalLM, _AutoTokenizer, _BitsAndBytesConfig, _PeftModel
    if _torch is None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            from peft import PeftModel
            _torch = torch
            _AutoModelForCausalLM = AutoModelForCausalLM
            _AutoTokenizer = AutoTokenizer
            _BitsAndBytesConfig = BitsAndBytesConfig
            _PeftModel = PeftModel
        except ImportError as e:
            raise ImportError(
                f"Local model inference requires torch / transformers / peft: {e}\n"
                "Install: pip install torch transformers peft bitsandbytes accelerate"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction (same schema as SemanticAgent)
# ──────────────────────────────────────────────────────────────────────────────
MATERIAL_EXTRACTION_SYSTEM_PROMPT = """You are a materials science literature knowledge extraction expert.
Your task is to precisely extract: material composition, processing, microstructure,
mechanical properties, and their causal relationships from materials science papers.
You must strictly follow the specified JSON schema output without any explanatory text."""


def _build_extraction_prompt(pages: List[Dict[str, Any]]) -> str:
    """Build the material semantic extraction prompt for local model.
    Same schema and constraints as SemanticAgent._build_prompt,
    but split into system / user messages for chat_template compatibility.
    """
    # Collect figure info from VisionAgent output
    figure_items = []
    for page in pages:
        imgs = page.get("images", [])
        caps = page.get("captions", [])
        rel = page.get("relations", {})
        for r in rel.get("image_captions", []):
            img_id = r["image_id"]
            cap_id = r["caption_id"]
            img = next((x for x in imgs if x["id"] == img_id), None)
            cap = next((x for x in caps if x["id"] == cap_id), None)
            if img:
                figure_items.append({
                    "figure_id": img["id"],
                    "caption_text": cap["text"] if cap else "",
                    "page": page.get("page_index", -1)
                })

    schema_block = """
Please read the following paper text and complete material causal knowledge graph information extraction.

Your tasks:
1. Identify all distinct material samples in the text
2. Build composition -> process -> microstructure -> property causal chains centered on each Sample
3. Only generate causal relations when explicit causal statements appear in the text
4. Output must be strictly valid JSON with no explanatory text

Extraction principles (must follow):
- Sample is the unique central node; do not merge different samples
- Prefer original wording from the text; avoid summary rewrites
- Use null for uncertain or missing information
- Do not establish cross-sample causal relations
- Do not supplement causality based on common knowledge

Output JSON format (strictly follow this schema):
{
  "samples": [
    {
      "id": "S1",
      "composition": {
        "canonical_name": "Ti-6Al-4V",
        "system": "Ti-Al-V",
        "elements_str": "Ti, Al, V",
        "elements": [
          { "element": "Ti", "content": "balance" },
          { "element": "Al", "content": "6 wt.%" },
          { "element": "V", "content": "4 wt.%" }
        ],
        "notes": null
      },
      "process": {
        "method": "heat treatment",
        "parameters": null,
        "temperature": "950 C",
        "duration": "2 h",
        "environment": "air",
        "notes": null
      },
      "microstructure": {
        "phase_type": ["alpha", "beta"],
        "morphology": "lamellar",
        "grain_size": null,
        "distribution": "uniform",
        "observation_method": "SEM",
        "figure_reference": ["Figure 3b"]
      },
      "property": {
        "type": "tensile strength",
        "value": "950",
        "units": "MPa",
        "test_condition": "room temperature",
        "notes": null
      },
      "causal_relations": [
        {
          "cause_node": "process.temperature(950C)",
          "effect_node": "microstructure.grain_size",
          "relation_type": "causal",
          "direction": "process_to_microstructure",
          "trigger_phrase": "led to",
          "source_sentence": "The increased heat treatment temperature led to grain refinement.",
          "polarity": "decrease"
        }
      ]
    }
  ],
  "causal_claims": [
    {
      "cause": "process.temperature(950C)",
      "cause_type": "process",
      "effect": "microstructure.grain_size",
      "effect_type": "microstructure",
      "polarity": "decrease",
      "claim_type": "explicit_causal",
      "confidence": 0.85,
      "evidence_text": "The increased heat treatment temperature led to grain refinement.",
      "evidence_ids": ["S1"]
    }
  ],
  "contrast_pairs": [],
  "mechanism_links": [],
  "global_figures": [
    {
      "figure_id": "Figure 3b",
      "caption_mention": "SEM images showing lamellar alpha/beta microstructure",
      "related_sample_ids": ["S1"]
    }
  ]
}

Causal node naming rules (CRITICAL, must follow):
cause and effect must use controlled path format:
  - composition.element(Ti) / composition.content(Al) / composition.ratio(Ti/Al)
  - process.method(heat treatment) / process.temperature(950C) / process.duration(2h)
  - microstructure.phase_type(alpha+beta) / microstructure.morphology(lamellar) / microstructure.grain_size
  - property.tensile_strength / property.yield_strength / property.elongation / property.hardness

Allowed causal directions (only these):
  composition->process / composition->microstructure / process->microstructure
  microstructure->property / process->property

Common terminology:
Process: VIM, VAR, ESR, HIP, solution treatment, aging, quenching, forging, SLM, EBM, DED
Phases: gamma, gamma', gamma'', delta, MC, M23C6, M6C, Laves, sigma, mu, eta, TCP
Properties: UTS, YS, EL, HV, creep rupture life, LCF life
"""

    figure_info = ""
    if figure_items:
        figure_info = "\nFigure information parsed from PDF (for figure_reference inference):\n"
        for fig in figure_items:
            cap_txt = fig["caption_text"].replace("\n", " ").strip()
            figure_info += f"- Image ID {fig['figure_id']} (page {fig['page']}): {cap_txt}\n"

    return schema_block + figure_info + "\nExecute material semantic extraction on the following text:\n\n{TEXT_BLOCK}\n"


class LocalLLMAgent:
    """Local Qwen2.5-7B-Instruct + LoRA fine-tuned model inference Agent.
    Interface compatible with SemanticAgent (both implement extract method).

    Parameters:
        base_model_path: Path to base model (local directory)
        lora_path:       LoRA adapter path (checkpoint directory)
        use_4bit:         Enable 4-bit BitsAndBytes quantization (recommended, saves VRAM)
        max_new_tokens:   Max generation tokens
        temperature:      Sampling temperature (0.1 for stability)
        device_map:       Device mapping ("auto" for automatic GPU/CPU selection)
    """

    def __init__(
        self,
        base_model_path: str,
        lora_path: str,
        use_4bit: bool = True,
        max_new_tokens: int = 4096,
        temperature: float = 0.1,
        device_map: str = "auto",
    ):
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.use_4bit = use_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device_map = device_map

        self.model = None
        self.tokenizer = None
        self._loaded = False

    # ──────────────────────────────────────────────────────────
    # Model loading
    # ──────────────────────────────────────────────────────────

    def load(self):
        """Explicitly load model (lazy, auto-called on first extract)."""
        if self._loaded:
            return

        _lazy_import()
        logger.info(f"[LocalLLMAgent] Loading base model: {self.base_model_path}")
        logger.info(f"[LocalLLMAgent] LoRA path: {self.lora_path}")

        # Path validation
        if not os.path.exists(self.base_model_path):
            raise FileNotFoundError(f"Base model path not found: {self.base_model_path}")
        if not os.path.exists(self.lora_path):
            raise FileNotFoundError(f"LoRA path not found: {self.lora_path}")
        adapter_cfg = os.path.join(self.lora_path, "adapter_config.json")
        if not os.path.exists(adapter_cfg):
            raise FileNotFoundError(f"adapter_config.json not found: {adapter_cfg}")

        # Quantization config
        bnb_config = None
        if self.use_4bit:
            bnb_config = _BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_torch.bfloat16,
            )

        # Load base model
        self.model = _AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=bnb_config,
            device_map=self.device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.info("[LocalLLMAgent] Base model loaded successfully")

        # Attach LoRA weights
        self.model = _PeftModel.from_pretrained(self.model, self.lora_path)
        logger.info("[LocalLLMAgent] LoRA weights loaded successfully")

        # Load tokenizer
        self.tokenizer = _AutoTokenizer.from_pretrained(
            self.base_model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("[LocalLLMAgent] Tokenizer loaded successfully")

        self._loaded = True

    def unload(self):
        """Release model VRAM (optional)."""
        if self._loaded:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._loaded = False
            if _torch is not None:
                _torch.cuda.empty_cache()
            logger.info("[LocalLLMAgent] Model unloaded")

    # ──────────────────────────────────────────────────────────
    # Core inference
    # ──────────────────────────────────────────────────────────

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Call local model to generate response."""
        _lazy_import()
        if not self._loaded:
            self.load()

        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if hasattr(inputs, "input_ids"):
                input_ids = inputs.input_ids.to(self.model.device)
            else:
                input_ids = inputs.to(self.model.device)

            with _torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=0.95,
                    repetition_penalty=1.05,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                )

            response = self.tokenizer.decode(
                output[0][input_ids.shape[1]:], skip_special_tokens=True
            )
            return response

        except Exception as e:
            logger.error(f"[LocalLLMAgent] Generation failed: {e}")
            raise

    def _clean_json(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from model output."""
        raw = text.strip()
        cleaned = raw

        if "```" in raw:
            m = re.search(r"```json(.*?)```", raw, re.S | re.I)
            if not m:
                m = re.search(r"```(.*?)```", raw, re.S)
            if m:
                cleaned = m.group(1).strip()

        try:
            return json.loads(cleaned)
        except Exception:
            pass

        try:
            fixed = cleaned.replace("'", '"')
            fixed = fixed.replace(",}", "}")
            fixed = fixed.replace(",]", "]")
            return json.loads(fixed)
        except Exception:
            pass

        # Try extracting outermost { ... }
        brace_match = re.search(r"\{.*\}", cleaned, re.S)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except Exception:
                pass

        logger.warning("[LocalLLMAgent] Model output is not valid JSON, returning raw_text")
        return {"raw_text": raw}

    # ──────────────────────────────────────────────────────────
    # Public interface (compatible with SemanticAgent.extract)
    # ──────────────────────────────────────────────────────────

    def extract(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Main entry point: semantic extraction from VisionAgent output pages.

        Returns:
            {
                "success": True,
                "semantic": { ... },   # Same schema as SemanticAgent
                "raw_text": "...",
                "source": "local_model"
            }
        """
        prompt_template = _build_extraction_prompt(pages)

        # Concatenate text
        full_text = ""
        for p in pages:
            if "text" in p:
                full_text += p["text"] + "\n"

        user_prompt = prompt_template.replace("{TEXT_BLOCK}", full_text)

        messages = [
            {"role": "system", "content": MATERIAL_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("[LocalLLMAgent] Starting local inference (Qwen2.5-7B-Instruct + LoRA)...")
        try:
            raw_output = self._generate(messages)
        except Exception as e:
            return {"error": f"Local model inference failed: {e}"}

        parsed = self._clean_json(raw_output)

        return {
            "success": True,
            "semantic": parsed,
            "raw_text": raw_output,
            "source": "local_model",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Factory: build LocalLLMAgent from config dict or defaults
# ──────────────────────────────────────────────────────────────────────────────

def build_local_agent(config: Optional[Dict[str, Any]] = None) -> "LocalLLMAgent":
    """Build a LocalLLMAgent instance from a config dict.
    If config is None, use default paths (consistent with training project).

    config example:
        {
            "base_model_path": "D:/models/Qwen2.5-7B-Instruct",
            "lora_path": "D:/python/Litextract_segmantic_data/lora_out/lora_out_qwen/checkpoint-1632",
            "use_4bit": true,
            "max_new_tokens": 4096,
            "temperature": 0.1
        }
    """
    if config is None:
        config = {
            "base_model_path": "D:/python/Litextract_segmantic_data/Qwen2.5-7B-Instruct",
            "lora_path": "D:/python/Litextract_segmantic_data/lora_out/lora_out_qwen/checkpoint-1632",
            "use_4bit": True,
            "max_new_tokens": 4096,
            "temperature": 0.1,
        }

    return LocalLLMAgent(
        base_model_path=config["base_model_path"],
        lora_path=config["lora_path"],
        use_4bit=config.get("use_4bit", True),
        max_new_tokens=config.get("max_new_tokens", 4096),
        temperature=config.get("temperature", 0.1),
    )

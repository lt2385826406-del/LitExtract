"""
"""

import json
import logging
import re
import os
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

#
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
                "Please run: pip install torch transformers peft bitsandbytes accelerate"
            )


# ================================================================
#
# ================================================================

MATERIAL_EXTRACTION_SYSTEM_PROMPT = """你是材料科学文献知识抽取专家。
你的任务是从材料科学论文中精确抽取：材料成分、加工工艺、microstructure、mechanical properties，以及它们之间的因果关系。
你必须严格遵循指定的 JSON schema output，不得添加任何解释性文字。"""


def _build_extraction_prompt(pages: List[Dict[str, Any]]) -> str:
    """Module functionality."""
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
                    "page": page.get("page_index", -1),
                })

    schema_block = """
请阅读以下论文段落，completed【材料因果knowledge graph】的info抽取。

你的任务是：
1. 识别文中所有不同的材料样品（Sample）
2. 以 Sample 为中心，build composition → process → microstructure → property 的因果链
3. 仅在文中出现"明确因果表述"时，才允许生成因果关系
4. 将 microstructure 与对应的image编号（如 Figure 3b, Fig.4）进行绑定
5. output必须是严格合法 JSON，不得包含任何解释性文字

抽取原则（必须遵守）：
- Sample 是唯一中心节点，不得merge不同样品
- 所有字段优先使用"原文原词"，避免总结性改写
- 不确定或未出现的info → 使用 null
- 不允许跨样品建立因果关系
- 不允许基于常识或背景知识补充因果

output JSON 格式（严格遵守此 schema）：
{
  "samples": [
    {
      "id": "S1",
      "composition": {
        "canonical_name": "IN718",
        "system": "Ni-Cr-Fe",
        "elements_str": "Ni-19Cr-19Fe-5Nb-3Mo",
        "elements": [
          {"element": "Ni", "content": "bal."},
          {"element": "Cr", "content": "19 wt%"}
        ],
        "notes": null
      },
      "processes": [
        {
          "step": 1,
          "method": "VIM",
          "temperature": null,
          "duration": null,
          "environment": "vacuum",
          "parameters": null,
          "notes": null
        }
      ],
      "microstructures": [
        {
          "phase_type": ["γ", "γ'"],
          "morphology": "cuboidal",
          "grain_size": "~0.5μm",
          "distribution": "uniformly distributed",
          "observation_method": "SEM",
          "figure_reference": ["Fig.3"]
        }
      ],
      "properties": [
        {
          "type": "UTS",
          "value": "1100",
          "units": "MPa",
          "test_condition": "25°C",
          "notes": null
        }
      ],
      "causal_relations": [
        {
          "cause": "aging at 845°C",
          "effect": "γ' precipitation strengthening",
          "trigger_word": "led to",
          "confidence": "high"
        }
      ]
    }
  ],
  "causal_claims": [],
  "contrast_pairs": [],
  "mechanism_links": [],
  "global_figures": []
}

常见术语Reference：
工艺: VIM, VAR, ESR, HIP, solution treatment, aging, quenching, forging, SLM, EBM, DED
相: γ, γ', γ'', δ, MC, M23C6, M6C, Laves, σ, μ, η, TCP, boride, carbide
性能: UTS, YS, EL, RA, HV, creep rupture life, LCF life, fatigue limit, CTE
"""

    figure_info = ""
    if figure_items:
    """
    Materials science literature semantic extraction Agent.
    Uses local Mistral-7B-Instruct + LoRA inference only.

    Args:
        base_model_path: Mistral-7B base model path
        lora_path:       LoRA adapter checkpoint path
        use_4bit:        Enable 4-bit quantization (recommended)
        max_new_tokens:  Maximum generated token count
        temperature:     Sampling temperature (0.1 for stability)
    """

    def __init__(
        self,
        base_model_path: str,
        lora_path: str,
        use_4bit: bool = True,
        max_new_tokens: int = 4096,
        temperature: float = 0.1,
    ):
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.use_4bit = use_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.model = None
        self.tokenizer = None
        self._loaded = False

    # # ── Model Loading ──

    def load(self):
        """
        if self._loaded:
            return

        _lazy_import()
        logger.info(f"[SemanticAgent] Loading base model: {self.base_model_path}")
        logger.info(f"[SemanticAgent] LoRA path: {self.lora_path}")

        if not os.path.exists(self.base_model_path):
            raise FileNotFoundError(f"Base model path not found: {self.base_model_path}")
        if not os.path.exists(self.lora_path):
            raise FileNotFoundError(f"LoRA path not found: {self.lora_path}")
        adapter_cfg = os.path.join(self.lora_path, "adapter_config.json")
        if not os.path.exists(adapter_cfg):
            raise FileNotFoundError(f"adapter_config.json not found: {adapter_cfg}")

        bnb_config = None
        if self.use_4bit:
            bnb_config = _BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=_torch.bfloat16,
            )

        self.model = _AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.info("[SemanticAgent] Base model loaded successfully")

        self.model = _PeftModel.from_pretrained(self.model, self.lora_path)
        logger.info("[SemanticAgent] LoRA weights loaded successfully")

        self.tokenizer = _AutoTokenizer.from_pretrained(
            self.base_model_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        logger.info("[SemanticAgent] Tokenizer loaded successfully")

        self._loaded = True

    def unload(self):
        """Module functionality."""
        if self._loaded:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self._loaded = False
            if _torch is not None:
                _torch.cuda.empty_cache()
            logger.info("[SemanticAgent] Model unloaded")

    # # ── Inference ──

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        """Module functionality."""
        _lazy_import()
        if not self._loaded:
            self.load()

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

        return self.tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        )

    def _clean_json(self, text: str) -> Dict[str, Any]:
        """Module functionality."""
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

        brace_match = re.search(r"\{.*\}", cleaned, re.S)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except Exception:
                pass

        logger.warning("[SemanticAgent] Model output is not valid JSON, returning raw_text")
        return {"raw_text": raw}

    #

    def extract(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        """
        prompt_template = _build_extraction_prompt(pages)

        full_text = ""
        for p in pages:
            if "text" in p:
                full_text += p["text"] + "\n"

        user_prompt = prompt_template.replace("{TEXT_BLOCK}", full_text)

        messages = [
            {"role": "system", "content": MATERIAL_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("[SemanticAgent] Starting local inference (Mistral-7B + LoRA)...")
        try:
            raw_output = self._generate(messages)
        except Exception as e:
            return {"error": f"Local model inference failed: {e}", "success": False}

        parsed = self._clean_json(raw_output)

        return {
            "success": True,
            "semantic": parsed,
            "raw_text": raw_output,
            "source": "mistral_7b_lora",
        }

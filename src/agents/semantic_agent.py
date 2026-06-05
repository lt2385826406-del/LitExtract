# agents/semantic_agent.py

import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class SemanticAgent:
    """
    Semantic extraction agent (local model mode).

    Uses Qwen2.5-7B-Instruct + LoRA fine-tuned weights for local inference.

    Returns:
      {
        "success": True,
        "semantic": { <materials knowledge JSON> },
        "raw_text": "...",
        "source": "local_model"
      }
    """

    def __init__(self, local_model_config: Optional[Dict[str, Any]] = None):
        """
        Args:
            local_model_config: Local model config dict, see LocalLLMAgent.build_local_agent
        """
        self.local_model_config = local_model_config

        # Lazy-init local model (avoid long startup load)
        self._local_agent = None

    # ---------------------------------------------------------

    def _get_local_agent(self):
        """Lazy-load local model agent."""
        if self._local_agent is None:
            from .local_llm_agent import build_local_agent
            self._local_agent = build_local_agent(self.local_model_config)
        return self._local_agent

    # ---------------------------------------------------------

    def extract(self, pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Entry point: run local model inference."""
        try:
            agent = self._get_local_agent()
            result = agent.extract(pages)
            return result
        except Exception as e:
            logger.error(f"[SemanticAgent] Local model inference failed: {e}")
            return {"error": f"Local model inference failed: {e}"}

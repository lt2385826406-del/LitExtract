# agents/coordinator_agent.py

import logging
import uuid
import os
from typing import Dict, Any, List, Optional

from agents.semantic_agent import SemanticAgent
from agents.knowledge_agent import KnowledgeAgent
from agents.kg_visualizer import generate_all_outputs

# VisionAgent is optional — graceful fallback if not importable
try:
    from agents.vision_agent import VisionAgent
except (ImportError, FileNotFoundError):
    VisionAgent = None

logger = logging.getLogger(__name__)


class CoordinatorAgent:
    """
    """

    def __init__(
        self,
        use_local_model: bool = False,
        local_model_config: Optional[Dict[str, Any]] = None,
    ):
        self.use_local_model = use_local_model
        self.local_model_config = local_model_config

        self.vision_agent = None
        self.semantic_agent = SemanticAgent(
            use_local_model=use_local_model,
            local_model_config=local_model_config,
        )
        self.knowledge_agent = KnowledgeAgent()

        # Try to initialize VisionAgent only if module is available
        if VisionAgent is not None:
            try:
                self.vision_agent = VisionAgent()
            except Exception as e:
                logger.warning(f"VisionAgent initialization failed: {e}. Continuing without VisionAgent.")
        else:
            self.vision_agent = None
            logger.info("VisionAgent module not available (utils.utils dependency). Use pdf_parser.VisionAgent instead.")

    # ---------------------------------------------------------

    def run_vision(self, pdf_path: str) -> List[Dict[str, Any]]:
        logger.info("Coordinator: Calling VisionAgent to parse PDF: %s", pdf_path)
        if self.vision_agent is None:
            logger.warning("VisionAgent not initialized, cannot run image detection")
            return []
        pages = self.vision_agent.run(pdf_path)
        logger.info("VisionAgent completed, %d pages total", len(pages))
        return pages

    # ---------------------------------------------------------

    def run_semantic(self, pdf_path: str, vision_pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        """
        logger.info(
            "Coordinator: Calling SemanticAgent for semantic extraction (local model)"
        )
        result = self.semantic_agent.extract(vision_pages)

        #
        if isinstance(result, dict) and "semantic" in result:
            return result["semantic"]
        return result

    # ---------------------------------------------------------

    def run_knowledge_graph(
        self,
        semantic_data: Dict[str, Any],
        pdf_path: str,
        output_dir: str = "outputs",
    ) -> Dict[str, Any]:
        """
        """
        logger.info("Coordinator: Calling KnowledgeAgent to build knowledge graph")
        paper_id = str(uuid.uuid4())

        # 1. Build knowledge graph
        kg_result = self.knowledge_agent.build_kg(semantic_data, paper_id, pdf_path)
        if not kg_result.get("success", False):
            return kg_result

        # 2. Build causal hypothesis graph
        chg_result = self.knowledge_agent.build_causal_graph(semantic_data, paper_id)
        if not chg_result.get("success", False):
            return chg_result

        kg_data = kg_result.get("knowledge_graph", {})
        chg_data = chg_result.get("graph", {})

        #
        prefix = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            vis_paths = generate_all_outputs(kg_data, chg_data, output_dir, prefix)
        except Exception as e:
            logger.warning(f"Coordinator: visualization generation failed (graph data unaffected): {e}")
            vis_paths = {}

        return {
            "success": True,
            "paper_id": paper_id,
            "message": "Knowledge graph and causal hypothesis graph built successfully",
            "knowledge_graph": kg_data,
            "causal_graph": chg_data,
            "visualization": vis_paths,
        }

    def save_knowledge_graph(self, file_path: str, format: str = "json") -> Dict[str, Any]:
        """Module functionality."""
        logger.info(f"Coordinator: saving knowledge graph to file {file_path}")
        return self.knowledge_agent.save_to_file(file_path, format)

    def run_causal_graph(self, semantic_data: Dict[str, Any], pdf_path: str) -> Dict[str, Any]:
        """Module functionality."""
        logger.info("Coordinator: Calling KnowledgeAgent to build causal hypothesis graph")
        paper_id = str(uuid.uuid4())
        chg_result = self.knowledge_agent.build_causal_graph(semantic_data, paper_id)
        if not chg_result.get("success", False):
            return chg_result
        return {"success": True, "paper_id": paper_id, "graph": chg_result.get("graph", {})}

    def run_full_pipeline(self, pdf_path: str) -> Dict[str, Any]:
        """Module functionality."""
        pages = self.run_vision(pdf_path)
        semantic = self.run_semantic(pdf_path, pages)

        if isinstance(semantic, dict) and "samples" in semantic:
            kg_result = self.run_knowledge_graph(semantic, pdf_path)
            return {"vision": pages, "semantic": semantic, "knowledge_graph": kg_result}

        return {"vision": pages, "semantic": semantic}


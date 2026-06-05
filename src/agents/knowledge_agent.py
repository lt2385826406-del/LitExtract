from typing import Dict, List, Any, Optional, Tuple
import re
from .kg_builder import KGBuilder
from .kg_schema import Evidence
from .causal_builder import CausalHypothesisGraph
from .vocab import (
    map_property_type,
    map_process_method,
    map_phase_type,
    map_morphology,
    map_polarity
)
import json
import logging

logger = logging.getLogger(__name__)

class KnowledgeAgent:
    """
    """
    
    def __init__(self):
        self.kg_builder = KGBuilder()
        self.causal_builder = CausalHypothesisGraph()
    
    @staticmethod
    def split_value_units(value_str: str) -> Tuple[str, str]:
        """
        """
        if not value_str:
            return "", ""
        
        #
        value_str = value_str.strip()
        
        #
        #
        #
        #
        match = re.match(r'^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)(.*)$', value_str)
        if match:
            value = match.group(1).strip()
            units = match.group(2).strip()
            return value, units
        
        #
        return value_str, ""
    
    def build_kg(self, semantic_data: Dict[str, Any], paper_id: str, doc_path: str) -> Dict[str, Any]:
        """
        """
        if not semantic_data or "samples" not in semantic_data:
            return {"success": False, "error": "No semantic data provided"}
        
        try:
            #
            paper_node = self.kg_builder.upsert_node(
                "Paper",
                key=paper_id,
                name=doc_path.split("/")[-1],
                doc_path=doc_path
            )
            
            #
            for sample in semantic_data["samples"]:
                sample_id = sample.get("id", f"sample_{hash(str(sample))}")
                sample_key = f"{paper_id}_{sample_id}"
                
                #
                sample_node = self.kg_builder.upsert_node(
                    "Sample",
                    key=sample_key,
                    name=f"Sample_{sample_id}",
                    paper_id=paper_id,
                    sample_id=sample_id
                )
                
                #
                #
                composition = sample.get("composition", {})
                alloy_canonical_name = composition.get("canonical_name", "")
                alloy_system = composition.get("system", "")
                
                #
                if alloy_canonical_name and alloy_canonical_name != "null":
                    alloy_key = alloy_canonical_name
                elif alloy_system and alloy_system != "null":
                    alloy_key = alloy_system
                else:
                    alloy_key = f"{paper_id}_{sample_id}"  
                
                #
                alloy_name = f"Unknown_Alloy_{sample_id}"
                
                #
                alloy_node = self.kg_builder.upsert_node(
                    "Alloy", 
                    key=alloy_key, 
                    name=alloy_name,
                    notes="Auto-created default alloy node (empty composition)"
                )
                
                #
                #
                source_sentence = sample.get("source_sentence", "")
                ev_text = source_sentence or f"Sample {sample_id} uses alloy {alloy_name}"
                ev_sample_alloy = self.kg_builder.add_evidence(
                    paper_id=paper_id,
                    doc_path=doc_path,
                    evidence_type="paragraph",
                    source_info={"sample_id": sample_id},
                    content=ev_text,
                    confidence=0.95
                )
                self.kg_builder.add_edge(
                    "HAS_ALLOY",
                    sample_node,
                    alloy_node,
                    confidence=0.95,
                    evidence_ids=[ev_sample_alloy.evidence_id]
                )
                
                #
                composition = sample.get("composition", {})
                if composition and composition != "null":
                    #
                    #
                    new_alloy_name = composition.get("canonical_name", "")
                    if not new_alloy_name or new_alloy_name == "null":
                        #
                        new_alloy_name = composition.get("elements_str", "")
                        if not new_alloy_name or new_alloy_name == "null":
                            #
                            elements_data = composition.get("elements", [])
                            if isinstance(elements_data, list) and len(elements_data) > 0:
                                element_names = [el.get("element", "").strip() for el in elements_data if isinstance(el, dict) and el.get("element")]
                                new_alloy_name = ", ".join(element_names)
                            else:
                                new_alloy_name = alloy_name  
                    
                    #
                    alloy_node = self.kg_builder.upsert_node(
                        "Alloy", 
                        key=alloy_key, 
                        name=new_alloy_name,
                        canonical_name=composition.get("canonical_name", ""),
                        system=composition.get("system", ""),
                        elements_str=composition.get("elements_str", ""),
                        elements=composition.get("elements", []),
                        notes=composition.get("notes")
                    )
                    alloy_name = new_alloy_name  
                    
                    #
                    composition_key = f"{paper_id}_{sample_id}_composition"
                    composition_node = self.kg_builder.upsert_node(
                        "Composition",
                        key=composition_key,
                        name=f"Composition of {alloy_name}",
                        canonical_name=composition.get("canonical_name", ""),
                        system=composition.get("system", ""),
                        elements_str=composition.get("elements_str", ""),
                        elements=composition.get("elements", []),
                        notes=composition.get("notes")
                    )
                    
                    #
                    #
                    composition_source_sentence = composition.get("source_sentence", "") or sample.get("source_sentence", "")
                    ev_text = composition_source_sentence or f"Composition of {alloy_name}: {composition.get('elements_str', alloy_name)}"
                    ev = self.kg_builder.add_evidence(
                        paper_id=paper_id,
                        doc_path=doc_path,
                        evidence_type="paragraph",
                        source_info={"sample_id": sample_id},
                        content=ev_text,
                        confidence=0.9
                    )
                    self.kg_builder.add_edge(
                        "HAS_COMPOSITION",
                        alloy_node,
                        composition_node,
                        confidence=0.9,
                        evidence_ids=[ev.evidence_id]
                    )
                    
                    #
                    #
                    elements_data = composition.get("elements", [])
                    elements_str = composition.get("elements_str", "")
                    
                    if isinstance(elements_data, list) and len(elements_data) > 0:
                        #
                        for el_item in elements_data:
                            if isinstance(el_item, dict) and "element" in el_item:
                                el_val = el_item.get("element")
                                if el_val is None or not isinstance(el_val, str):
                                    continue
                                element = el_val.strip()
                                if element:
                                    element_node = self.kg_builder.upsert_node(
                                        "Element",
                                        key=element,
                                        name=element,
                                        content=el_item.get("content", "")
                                    )
                                    #
                                    #
                                    element_source_sentence = el_item.get("source_sentence", "") or composition_source_sentence
                                    ev_text = element_source_sentence or f"{alloy_name} contains {element}"
                                    ev = self.kg_builder.add_evidence(
                                        paper_id=paper_id,
                                        doc_path=doc_path,
                                        evidence_type="paragraph",
                                        source_info={"sample_id": sample_id},
                                        content=ev_text,
                                        confidence=0.95
                                    )
                                    self.kg_builder.add_edge(
                                        "HAS_ELEMENT",
                                        alloy_node,
                                        element_node,
                                        confidence=0.95,
                                        evidence_ids=[ev.evidence_id]
                                    )
                    elif elements_str and elements_str != "null":
                        #
                        elements = [el.strip() for el in elements_str.split(",") if el.strip()]
                        for element in elements:
                            element_node = self.kg_builder.upsert_node(
                                "Element",
                                key=element,
                                name=element
                            )
                            #
                            #
                            ev_text = composition_source_sentence or f"{alloy_name} contains {element}"
                            ev = self.kg_builder.add_evidence(
                                paper_id=paper_id,
                                doc_path=doc_path,
                                evidence_type="paragraph",
                                source_info={"sample_id": sample_id},
                                content=ev_text,
                                confidence=0.95
                            )
                            self.kg_builder.add_edge(
                                "HAS_ELEMENT",
                                alloy_node,
                                element_node,
                                confidence=0.95,
                                evidence_ids=[ev.evidence_id]
                            )
                
                #
                processes = sample.get("processes", sample.get("process", []))
                if isinstance(processes, dict) and processes and processes != "null":
                    processes = [processes]  
                if isinstance(processes, list):
                    for pi, process in enumerate(processes):
                        if not process or process == "null":
                            continue
                        process_key = f"{paper_id}_{sample_id}_process_{pi}"
                        raw_method = process.get("method", "Unknown Process")
                        process_name = map_process_method(raw_method)
                        process_node = self.kg_builder.upsert_node(
                            "Processing",
                            key=process_key,
                            name=process_name,
                            method=process_name,
                            raw_method=raw_method,
                            parameters=process.get("parameters"),
                            temperature=process.get("temperature"),
                            duration=process.get("duration"),
                            environment=process.get("environment"),
                            notes=process.get("notes")
                        )
                        
                        #
                        process_source_sentence = process.get("source_sentence", "") or sample.get("source_sentence", "")
                        ev_text = process_source_sentence or f"Sample {sample_id} ({alloy_name}) is processed by {process_name}"
                        ev = self.kg_builder.add_evidence(
                            paper_id=paper_id,
                            doc_path=doc_path,
                            evidence_type="paragraph",
                            source_info={"sample_id": sample_id, "process_index": pi},
                            content=ev_text,
                            confidence=0.9
                        )
                        self.kg_builder.add_edge(
                            "PROCESSED_BY",
                            sample_node,
                            process_node,
                            confidence=0.9,
                            evidence_ids=[ev.evidence_id]
                        )
                
                #
                microstructures = sample.get("microstructures", sample.get("microstructure", []))
                if isinstance(microstructures, dict) and microstructures and microstructures != "null":
                    microstructures = [microstructures]
                if isinstance(microstructures, list):
                    for mi, microstructure in enumerate(microstructures):
                        if not microstructure or microstructure == "null":
                            continue
                        micro_key = f"{paper_id}_{sample_id}_micro_{mi}"
                        raw_phase_type = microstructure.get("phase_type", "Unknown Microstructure")
                        raw_morphology = microstructure.get("morphology", "")
                        
                        #
                        if isinstance(raw_phase_type, list):
                            mapped_phase_types = [map_phase_type(pt) for pt in raw_phase_type]
                            phase_type = " + ".join(mapped_phase_types)
                            raw_phase_type_str = ", ".join([str(pt) for pt in raw_phase_type])
                        else:
                            phase_type = map_phase_type(raw_phase_type)
                            raw_phase_type_str = str(raw_phase_type)
                        
                        #
                        if isinstance(raw_morphology, list):
                            morphology = ", ".join([map_morphology(m) for m in raw_morphology])
                            raw_morphology_str = ", ".join([str(m) for m in raw_morphology])
                        else:
                            morphology = map_morphology(raw_morphology) if raw_morphology else ""
                            raw_morphology_str = str(raw_morphology) if raw_morphology else ""
                        
                        micro_name = phase_type
                        
                        micro_node = self.kg_builder.upsert_node(
                            "Microstructure",
                            key=micro_key,
                            name=micro_name,
                            phase_type=phase_type,
                            raw_phase_type=raw_phase_type_str,
                            morphology=morphology,
                            raw_morphology=raw_morphology_str,
                            grain_size=microstructure.get("grain_size"),
                            distribution=microstructure.get("distribution"),
                            observation_method=microstructure.get("observation_method"),
                            figure_reference=microstructure.get("figure_reference")
                        )
                        
                        #
                        micro_source_sentence = microstructure.get("source_sentence", "") or sample.get("source_sentence", "")
                        ev_text = micro_source_sentence or f"Sample {sample_id} ({alloy_name}) has microstructure: {micro_name}"
                        ev = self.kg_builder.add_evidence(
                            paper_id=paper_id,
                            doc_path=doc_path,
                            evidence_type="paragraph",
                            source_info={"sample_id": sample_id, "micro_index": mi},
                            content=ev_text,
                            confidence=0.85
                        )
                        self.kg_builder.add_edge(
                            "HAS_MICROSTRUCTURE",
                            sample_node,
                            micro_node,
                            confidence=0.85,
                            evidence_ids=[ev.evidence_id]
                        )
                        
                        #
                        if isinstance(processes, list) and len(processes) > 0:
                            for pj, _ in enumerate(processes):
                                proc_key = f"{paper_id}_{sample_id}_process_{pj}"
                                process_node = self.kg_builder.kg.get_node_by_key("Processing", proc_key)
                                if process_node:
                                    forms_source_sentence = micro_source_sentence or f"{process_node.name} forms microstructure {micro_name}"
                                    ev_forms = self.kg_builder.add_evidence(
                                        paper_id=paper_id,
                                        doc_path=doc_path,
                                        evidence_type="paragraph",
                                        source_info={"sample_id": sample_id, "process_index": pj, "micro_index": mi},
                                        content=forms_source_sentence,
                                        confidence=0.8
                                    )
                                    self.kg_builder.add_edge(
                                        "FORMS_MICROSTRUCTURE",
                                        process_node,
                                        micro_node,
                                        confidence=0.8,
                                        evidence_ids=[ev_forms.evidence_id]
                                    )
                
                #
                properties = sample.get("properties", sample.get("property", []))
                if isinstance(properties, dict) and properties and properties != "null":
                    properties = [properties]
                if isinstance(properties, list):
                    for p_idx, property_data in enumerate(properties):
                        if not property_data or property_data == "null":
                            continue
                        #
                        raw_prop_type = property_data.get("type", "Unknown Property")
                        prop_type = map_property_type(raw_prop_type)
                        prop_value = property_data.get("value", "")
                        prop_units = property_data.get("units", "")
                        
                        if isinstance(prop_value, str) and not prop_units:
                            prop_value, prop_units = self.split_value_units(prop_value)
                        
                        prop_key = f"{paper_id}_{sample_id}_prop_{p_idx}"
                        prop_name = prop_type
                        prop_node = self.kg_builder.upsert_node(
                            "Property",
                            key=prop_key,
                            name=prop_name,
                            type=prop_type,
                            raw_type=raw_prop_type,
                            value=prop_value,
                            units=prop_units,
                            test_condition=property_data.get("test_condition"),
                            notes=property_data.get("notes")
                        )
                        
                        prop_source_sentence = property_data.get("source_sentence", "") or sample.get("source_sentence", "")
                        ev_text = prop_source_sentence or f"Sample {sample_id} ({alloy_name}) has property: {prop_name} = {prop_value} {prop_units}"
                        ev = self.kg_builder.add_evidence(
                            paper_id=paper_id,
                            doc_path=doc_path,
                            evidence_type="paragraph",
                            source_info={"sample_id": sample_id, "property_index": p_idx},
                            content=ev_text,
                            confidence=0.85
                        )
                        self.kg_builder.add_edge(
                            "HAS_PROPERTY",
                            sample_node,
                            prop_node,
                            confidence=0.85,
                            value=prop_value,
                            unit=prop_units,
                            evidence_ids=[ev.evidence_id]
                        )
                        
                        #
                        if isinstance(microstructures, list) and len(microstructures) > 0:
                            for mj, _ in enumerate(microstructures):
                                micro_key_mj = f"{paper_id}_{sample_id}_micro_{mj}"
                                micro_node = self.kg_builder.kg.get_node_by_key("Microstructure", micro_key_mj)
                                if micro_node:
                                    affects_text = prop_source_sentence or f"Microstructure {micro_node.name} affects property {prop_name}"
                                    ev_aff = self.kg_builder.add_evidence(
                                        paper_id=paper_id,
                                        doc_path=doc_path,
                                        evidence_type="paragraph",
                                        source_info={"sample_id": sample_id, "micro_index": mj, "property_index": p_idx},
                                        content=affects_text,
                                        confidence=0.8
                                    )
                                    self.kg_builder.add_edge(
                                        "AFFECTS_PROPERTY",
                                        micro_node,
                                        prop_node,
                                        confidence=0.8,
                                        evidence_ids=[ev_aff.evidence_id]
                                    )
            
            #
            if "global_figures" in semantic_data:
                for global_fig in semantic_data["global_figures"]:
                    fig_key = global_fig.get("figure_id", f"fig_{hash(str(global_fig))}")
                    fig_id = global_fig.get("figure_id", "Unknown Figure")
                    caption_mention = global_fig.get("caption_mention", "")
                    related_sample_ids = global_fig.get("related_sample_ids", [])
                    
                    #
                    fig_node = self.kg_builder.upsert_node(
                        "Figure",
                        key=fig_key,
                        name=fig_id,
                        caption_mention=caption_mention,
                        related_sample_ids=related_sample_ids
                    )
                    
                    #
                    ev_text = f"Figure {fig_id} shows {caption_mention}"
                    ev = self.kg_builder.add_evidence(
                        paper_id=paper_id,
                        doc_path=doc_path,
                        evidence_type="figure",
                        source_info={"figure_id": fig_id},
                        content=ev_text,
                        confidence=0.9
                    )
                    self.kg_builder.add_edge(
                        "HAS_FIGURE",
                        paper_node,
                        fig_node,
                        confidence=0.9,
                        evidence_ids=[ev.evidence_id]
                    )
                    
                    #
                    if related_sample_ids:
                        for sample_id in related_sample_ids:
                            micro_key = f"{paper_id}_{sample_id}_micro"
                            micro_node = self.kg_builder.kg.get_node_by_key("Microstructure", micro_key)
                            if micro_node:
                                #
                                micro_name = micro_node.name
                                ev_text_micro = f"Microstructure {micro_name} is shown in Figure {fig_id}: {caption_mention}"
                                ev_micro = self.kg_builder.add_evidence(
                                    paper_id=paper_id,
                                    doc_path=doc_path,
                                    evidence_type="figure",
                                    source_info={"figure_id": fig_id, "sample_id": sample_id},
                                    content=ev_text_micro,
                                    confidence=0.85
                                )
                                self.kg_builder.add_edge(
                                    "SHOWN_IN",
                                    micro_node,
                                    fig_node,
                                    confidence=0.85,
                                    evidence_ids=[ev_micro.evidence_id]
                                )
                                print(f"[DEBUG] Established Microstructure -> SHOWN_IN -> Figure edge: {micro_name} -> {fig_id}")
            
            #
            for sample in semantic_data["samples"]:
                sample_id = sample.get("id", f"sample_{hash(str(sample))}")
                sample_key = f"{paper_id}_{sample_id}"
                sample_node = self.kg_builder.kg.get_node_by_key("Sample", sample_key)
                if sample_node:
                    ev_text = f"Sample {sample_id} reported in {doc_path.split('/')[-1]}"
                    ev = self.kg_builder.add_evidence(
                        paper_id=paper_id,
                        doc_path=doc_path,
                        evidence_type="document",
                        source_info={"sample_id": sample_id},
                        content=ev_text,
                        confidence=0.95
                    )
                    self.kg_builder.add_edge(
                        "REPORTED_IN",
                        sample_node,
                        paper_node,
                        confidence=0.95,
                        evidence_ids=[ev.evidence_id]
                    )
            
            #
            kg_data = self.kg_builder.export_bundle(format="json")
            
            return {
                "success": True, 
                "message": "Knowledge graph built successfully",
                "knowledge_graph": kg_data
            }
        
        except Exception as e:
            logger.error(f"Error building knowledge graph: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def build_causal_graph(self, semantic_data: Dict[str, Any], paper_id: str) -> Dict[str, Any]:
        """
        """
        if not semantic_data:
            return {"success": False, "error": "No semantic data provided"}
        
        try:
            #
            self.causal_builder.clear()
            
            #
            causal_claims_from_samples = []
            if "samples" in semantic_data:
                for sample in semantic_data["samples"]:
                    sample_id = sample.get("id", "unknown")
                    causal_relations = sample.get("causal_relations", [])
                    
                    for relation in causal_relations:
                        if isinstance(relation, dict):
                            #
                            cause_node = relation.get("cause_node", "")
                            effect_node = relation.get("effect_node", "")
                            trigger_phrase = relation.get("trigger_phrase", "")
                            source_sentence = relation.get("source_sentence", "")
                            
                            #
                            polarity = relation.get("polarity", "")
                            if not polarity or polarity == "unknown":
                                #
                                polarity = "increase"
                                if trigger_phrase:
                                    decrease_phrases = ["decrease", "reduce", "suppress", "lower", "less", "refinement"]
                                    increase_phrases = ["increase", "enhance", "improve", "promote", "higher", "larger"]
                                    trigger_lower = trigger_phrase.lower()
                                    if any(p in trigger_lower for p in decrease_phrases):
                                        polarity = "decrease"
                                    elif any(p in trigger_lower for p in increase_phrases):
                                        polarity = "increase"
                            #
                            polarity = map_polarity(polarity)
                            
                            #
                            claim_type = relation.get("relation_type", "explicit_causal")
                            if claim_type == "causal":
                                claim_type = "explicit_causal"
                            
                            causal_claims_from_samples.append({
                                "cause": cause_node,
                                "effect": effect_node,
                                "polarity": polarity,
                                "claim_type": claim_type,
                                "confidence": 0.7,
                                "evidence_text": source_sentence,
                                "evidence_ids": [sample_id]
                            })
            
            if causal_claims_from_samples:
                logger.info(f"Bridged {len(causal_claims_from_samples)} causal claims from samples[].causal_relations")
                self.causal_builder.build_from_causal_claims(causal_claims_from_samples)
            
            #
            if "causal_claims" in semantic_data and semantic_data["causal_claims"]:
                logger.info(f"Processing {len(semantic_data['causal_claims'])} causal claims")
                self.causal_builder.build_from_causal_claims(semantic_data["causal_claims"])
            
            #
            if "contrast_pairs" in semantic_data and semantic_data["contrast_pairs"]:
                logger.info(f"Processing {len(semantic_data['contrast_pairs'])} contrast pairs")
                self.causal_builder.build_from_contrast_pairs(semantic_data["contrast_pairs"])
            
            #
            if "mechanism_links" in semantic_data and semantic_data["mechanism_links"]:
                logger.info(f"Processing {len(semantic_data['mechanism_links'])} mechanism links")
                self.causal_builder.build_from_mechanism_links(semantic_data["mechanism_links"])
            
            #
            chg_data = self.causal_builder.export_json()
            
            return {
                "success": True, 
                "message": "Causal hypothesis graph built successfully",
                "graph": chg_data
            }
            
        except Exception as e:
            logger.error(f"Error building causal graph: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def export_bundle(self, format: str = "json") -> Dict[str, Any]:
        """
        """
        try:
            data = self.kg_builder.export_bundle(format)
            return {
                "success": True,
                "format": format,
                "data": data
            }
        except Exception as e:
            logger.error(f"Error exporting knowledge graph: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def save_to_file(self, file_path: str, format: str = "json") -> Dict[str, Any]:
        """
        """
        try:
            self.kg_builder.save_to_file(file_path, format)
            return {
                "success": True,
                "file_path": file_path,
                "format": format
            }
        except Exception as e:
            logger.error(f"Error saving knowledge graph to file: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def load_from_file(self, file_path: str) -> Dict[str, Any]:
        """
        """
        try:
            self.kg_builder.load_from_file(file_path)
            return {
                "success": True,
                "file_path": file_path
            }
        except Exception as e:
            logger.error(f"Error loading knowledge graph from file: {e}")
            return {
                "success": False,
                "error": str(e)
            }

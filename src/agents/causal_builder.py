from typing import Dict, Any, List, Optional
from enum import Enum
import networkx as nx
import uuid


class ClaimType(str, Enum):
    EXPLICIT_CAUSAL = "explicit_causal"
    CONTRAST_BASED = "contrast_based"
    MECHANISM_SUPPORTED = "mechanism_supported"
    COOCCURRENCE = "cooccurrence"


class Polarity(str, Enum):
    PROMOTE = "promote"
    SUPPRESS = "suppress"
    INCREASE = "increase"
    DECREASE = "decrease"


class Strength(str, Enum):
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"


class CausalHypothesisGraph:
    def __init__(self):
        self.cg = nx.MultiDiGraph()
        self.evidence_map: Dict[str, List[str]] = {}
        self.node_attributes: Dict[str, Dict[str, Any]] = {}
    
    def _generate_edge_id(self, src: str, dst: str, polarity: str, claim_type: str) -> str:
        return f"{src}->{dst}:{polarity}:{claim_type}"
    
    def _calculate_strength(self, claim_type: str, confidence: float, evidence_count: int) -> str:
        base_score = confidence
        if claim_type == ClaimType.MECHANISM_SUPPORTED.value:
            base_score += 0.15
        elif claim_type == ClaimType.EXPLICIT_CAUSAL.value:
            base_score += 0.1
        elif claim_type == ClaimType.CONTRAST_BASED.value:
            base_score += 0.05
        
        evidence_multiplier = min(1.0 + (evidence_count * 0.1), 1.5)
        final_score = base_score * evidence_multiplier
        
        if final_score >= 0.8:
            return Strength.STRONG.value
        elif final_score >= 0.5:
            return Strength.MEDIUM.value
        else:
            return Strength.WEAK.value
    
    def add_causal_edge(self, src: str, dst: str, polarity: str, claim_type: str,
                        confidence: float, evidence_ids: List[str], 
                        evidence_text: str = ""):
        edge_id = self._generate_edge_id(src, dst, polarity, claim_type)
        
        if self.cg.has_edge(src, dst, key=edge_id):
            existing_edge = self.cg[src][dst][edge_id]
            existing_evidence = existing_edge.get("evidence_ids", [])
            combined_evidence = list(dict.fromkeys(existing_evidence + evidence_ids))
            new_confidence = (existing_edge.get("confidence", confidence) + confidence) / 2
            new_confidence = min(new_confidence * 1.1, 1.0)
            
            self.cg[src][dst][edge_id].update({
                "confidence": new_confidence,
                "evidence_ids": combined_evidence,
                "evidence_text": evidence_text if evidence_text else existing_edge.get("evidence_text", ""),
                "strength": self._calculate_strength(claim_type, new_confidence, len(combined_evidence))
            })
        else:
            strength = self._calculate_strength(claim_type, confidence, len(evidence_ids))
            self.cg.add_edge(src, dst, key=edge_id,
                           polarity=polarity,
                           claim_type=claim_type,
                           confidence=float(confidence),
                           strength=strength,
                           evidence_ids=list(dict.fromkeys(evidence_ids)),
                           evidence_text=evidence_text)
    
    def add_contrast_edge(self, varied: str, control: str, treatment: str, 
                         observed_change: str, claim_type: str = ClaimType.CONTRAST_BASED.value,
                         polarity: str = "increase", confidence: float = 0.7):
        varied_node = f"var:{varied}"
        control_node = f"ctrl:{control}"
        treatment_node = f"trt:{treatment}"
        
        self.add_causal_edge(control_node, varied_node, polarity, claim_type, 
                           confidence, [f"contrast:{varied}:{treatment}"], observed_change)
        
        if hasattr(self, 'contrast_info'):
            pass
        
        self.cg.nodes[varied_node].update({
            "node_type": "varied_factor",
            "original_varied": varied,
            "control_value": control,
            "treatment_value": treatment,
            "observed_change": observed_change
        })
    
    def add_mechanism_link(self, microstructure: str, property_name: str, 
                          mechanism: str, evidence_text: str = "",
                          claim_type: str = ClaimType.MECHANISM_SUPPORTED.value,
                          confidence: float = 0.75):
        ms_node = f"ms:{microstructure}"
        prop_node = f"prop:{property_name}"
        mech_node = f"mech:{mechanism}"
        
        self.add_causal_edge(ms_node, mech_node, "increase", claim_type, confidence,
                           [f"mechanism:{microstructure}:{mechanism}"], 
                           f"Microstructure '{microstructure}' involves mechanism '{mechanism}'")
        
        self.add_causal_edge(mech_node, prop_node, "increase", claim_type, confidence,
                           [f"mechanism:{mechanism}:{property_name}"], 
                           f"Mechanism '{mechanism}' affects property '{property_name}'")
    
    def add_node(self, node_id: str, node_type: str, attributes: Dict[str, Any] = None):
        if node_id not in self.cg.nodes:
            self.cg.add_node(node_id, node_type=node_type)
            if attributes:
                self.cg.nodes[node_id].update(attributes)
    
    def build_from_causal_claims(self, causal_claims: List[Dict[str, Any]], evidence_map: Dict[str, Any] = None):
        for claim in causal_claims:
            cause = claim.get("cause", "")
            effect = claim.get("effect", "")
            polarity = claim.get("polarity", "increase")
            claim_type = claim.get("claim_type", ClaimType.EXPLICIT_CAUSAL.value)
            confidence = claim.get("confidence", 0.7)
            evidence_text = claim.get("evidence_text", "")
            evidence_ids = claim.get("evidence_ids", [])
            
            if cause and effect:
                self.add_causal_edge(cause, effect, polarity, claim_type, 
                                   confidence, evidence_ids, evidence_text)
    
    def build_from_contrast_pairs(self, contrast_pairs: List[Dict[str, Any]]):
        for pair in contrast_pairs:
            varied = pair.get("varied", "")
            control = pair.get("control", "")
            treatment = pair.get("treatment", "")
            observed_change = pair.get("observed_change", "")
            
            if varied and (control or treatment):
                self.add_contrast_edge(varied, control, treatment, observed_change)
    
    def build_from_mechanism_links(self, mechanism_links: List[Dict[str, Any]]):
        for link in mechanism_links:
            ms = link.get("microstructure", "")
            prop = link.get("property", "")
            mechanism = link.get("mechanism", "")
            evidence_text = link.get("evidence_text", "")
            confidence = link.get("confidence", 0.75)
            
            if ms and mechanism:
                self.add_mechanism_link(ms, prop if prop else "", mechanism, evidence_text, confidence=confidence)
    
    def get_edge_data(self, src: str, dst: str) -> List[Dict[str, Any]]:
        edges = []
        if self.cg.has_edge(src, dst):
            for key, data in self.cg[src][dst].items():
                edge_info = {
                    "source": src,
                    "target": dst,
                    "polarity": data.get("polarity"),
                    "claim_type": data.get("claim_type"),
                    "confidence": data.get("confidence"),
                    "strength": data.get("strength"),
                    "evidence_ids": data.get("evidence_ids", []),
                    "evidence_text": data.get("evidence_text", "")
                }
                edges.append(edge_info)
        return edges
    
    def get_all_causal_edges(self) -> List[Dict[str, Any]]:
        edges = []
        for src, dst, key, data in self.cg.edges(data=True, keys=True):
            edge_info = {
                "source": src,
                "target": dst,
                "polarity": data.get("polarity"),
                "claim_type": data.get("claim_type"),
                "confidence": data.get("confidence"),
                "strength": data.get("strength"),
                "evidence_ids": data.get("evidence_ids", []),
                "evidence_text": data.get("evidence_text", "")
            }
            edges.append(edge_info)
        return edges
    
    def get_statistics(self) -> Dict[str, Any]:
        edges = self.get_all_causal_edges()
        claim_types = {}
        polarities = {}
        strengths = {}
        
        for edge in edges:
            ct = edge.get("claim_type", "unknown")
            pol = edge.get("polarity", "unknown")
            strg = edge.get("strength", "unknown")
            
            claim_types[ct] = claim_types.get(ct, 0) + 1
            polarities[pol] = polarities.get(pol, 0) + 1
            strengths[strg] = strengths.get(strg, 0) + 1
        
        return {
            "total_nodes": self.cg.number_of_nodes(),
            "total_edges": self.cg.number_of_edges(),
            "claim_type_distribution": claim_types,
            "polarity_distribution": polarities,
            "strength_distribution": strengths
        }
    
    def export_graphml(self, path: str):
        graphml_data = nx.write_graphml(self.cg, path)
        return graphml_data
    
    def export_json(self) -> Dict[str, Any]:
        return {
            "nodes": [{"id": n, "type": self.cg.nodes[n].get("node_type", "unknown"), 
                      "attributes": dict(self.cg.nodes[n])} 
                     for n in self.cg.nodes()],
            "edges": self.get_all_causal_edges(),
            "statistics": self.get_statistics()
        }
    
    def clear(self):
        self.cg.clear()
        self.evidence_map.clear()
        self.node_attributes.clear()

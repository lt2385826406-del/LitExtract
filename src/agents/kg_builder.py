from typing import Dict, List, Any, Optional
import uuid
import json
from datetime import datetime
from .kg_schema import Node, Edge, Evidence, KnowledgeGraph

class KGBuilder:
    def __init__(self):
        self.kg = KnowledgeGraph()
    
    def upsert_node(self, node_type: str, key: str, name: str, **properties) -> Node:
        """
        """
        existing_node = self.kg.get_node_by_key(node_type, key)
        if existing_node:
            #
            existing_node.properties.update(properties)
            existing_node.updated_at = datetime.now().isoformat()
            return existing_node
        else:
            #
            node_id = str(uuid.uuid4())
            node = Node(
                node_id=node_id,
                node_type=node_type,
                key=key,
                name=name,
                properties=properties
            )
            self.kg.add_node(node)
            return node
    
    def add_edge(self, relationship_type: str, source_node: Node, target_node: Node, 
                 confidence: float = 0.7, evidence_ids: List[str] = None, 
                 value: Any = None, unit: str = None, **properties) -> Edge:
        """
        """
        edge_id = str(uuid.uuid4())
        edge = Edge(
            edge_id=edge_id,
            relationship_type=relationship_type,
            source_node_id=source_node.node_id,
            target_node_id=target_node.node_id,
            evidence_ids=evidence_ids or [],
            confidence=confidence,
            value=value,
            unit=unit,
            properties=properties
        )
        self.kg.add_edge(edge)
        return edge
    
    def add_evidence(self, paper_id: str, doc_path: str, evidence_type: str, 
                    source_info: Dict[str, Any], content: str, 
                    confidence: float = 0.7) -> Evidence:
        """
        """
        evidence = Evidence.create(
            paper_id=paper_id,
            doc_path=doc_path,
            evidence_type=evidence_type,
            source_info=source_info,
            content=content,
            confidence=confidence
        )
        self.kg.add_evidence(evidence)
        return evidence
    
    def export_bundle(self, format: str = "json") -> Dict[str, Any] or str:
        """
        """
        if format == "json":
            return self._export_json()
        elif format == "graphml":
            return self._export_graphml()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_json(self) -> Dict[str, Any]:
        """
        """
        nodes = []
        for node in self.kg.nodes.values():
            nodes.append({
                "node_id": node.node_id,
                "node_type": node.node_type,
                "key": node.key,
                "name": node.name,
                "properties": node.properties,
                "created_at": node.created_at,
                "updated_at": node.updated_at
            })
        
        edges = []
        for edge in self.kg.edges:
            edges.append({
                "edge_id": edge.edge_id,
                "relationship_type": edge.relationship_type,
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "evidence_ids": edge.evidence_ids,
                "confidence": edge.confidence,
                "value": edge.value,
                "unit": edge.unit,
                "properties": edge.properties,
                "created_at": edge.created_at,
                "updated_at": edge.updated_at
            })
        
        evidences = []
        for evidence in self.kg.evidences.values():
            evidences.append({
                "evidence_id": evidence.evidence_id,
                "paper_id": evidence.paper_id,
                "doc_path": evidence.doc_path,
                "evidence_type": evidence.evidence_type,
                "source_info": evidence.source_info,
                "content": evidence.content,
                "confidence": evidence.confidence,
                "created_at": evidence.created_at
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "evidences": evidences
        }
    
    def _export_graphml(self) -> str:
        """
        """
        graphml = ['<?xml version="1.0" encoding="UTF-8"?>',
                  '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
                  '<graph id="knowledge_graph" edgedefault="directed">']
        
        #
        for node in self.kg.nodes.values():
            node_attrs = f'id="{node.node_id}" type="{node.node_type}" name="{node.name}" key="{node.key}"'
            for k, v in node.properties.items():
                node_attrs += f' {k}="{v}"'
            graphml.append(f'  <node {node_attrs}/>')
        
        #
        for edge in self.kg.edges:
            edge_attrs = f'id="{edge.edge_id}" source="{edge.source_node_id}" target="{edge.target_node_id}" '
            edge_attrs += f'relationship="{edge.relationship_type}" confidence="{edge.confidence}"'
            if edge.value is not None:
                edge_attrs += f' value="{edge.value}" unit="{edge.unit}"'
            for k, v in edge.properties.items():
                edge_attrs += f' {k}="{v}"'
            graphml.append(f'  <edge {edge_attrs}/>')
        
        graphml.extend(['</graph>', '</graphml>'])
        return '\n'.join(graphml)
    
    def save_to_file(self, file_path: str, format: str = "json"):
        """
        """
        data = self.export_bundle(format)
        if format == "json":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(data)
    
    def load_from_file(self, file_path: str):
        """
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        #
        for node_data in data['nodes']:
            node = Node(
                node_id=node_data['node_id'],
                node_type=node_data['node_type'],
                key=node_data['key'],
                name=node_data['name'],
                properties=node_data['properties'],
                created_at=node_data['created_at'],
                updated_at=node_data['updated_at']
            )
            self.kg.add_node(node)
        
        #
        for edge_data in data['edges']:
            edge = Edge(
                edge_id=edge_data['edge_id'],
                relationship_type=edge_data['relationship_type'],
                source_node_id=edge_data['source_node_id'],
                target_node_id=edge_data['target_node_id'],
                evidence_ids=edge_data['evidence_ids'],
                confidence=edge_data['confidence'],
                value=edge_data['value'],
                unit=edge_data['unit'],
                properties=edge_data['properties'],
                created_at=edge_data['created_at'],
                updated_at=edge_data['updated_at']
            )
            self.kg.add_edge(edge)
        
        #
        for evidence_data in data['evidences']:
            evidence = Evidence(
                evidence_id=evidence_data['evidence_id'],
                paper_id=evidence_data['paper_id'],
                doc_path=evidence_data['doc_path'],
                evidence_type=evidence_data['evidence_type'],
                source_info=evidence_data['source_info'],
                content=evidence_data['content'],
                confidence=evidence_data['confidence'],
                created_at=evidence_data['created_at']
            )
            self.kg.add_evidence(evidence)

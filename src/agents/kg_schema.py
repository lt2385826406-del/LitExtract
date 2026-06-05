from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import uuid
from datetime import datetime

@dataclass
class Node:
    node_id: str
    node_type: str
    key: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.node_id == other.node_id
        return False

    def __hash__(self):
        return hash(self.node_id)

@dataclass
class Evidence:
    evidence_id: str
    paper_id: str
    doc_path: str
    evidence_type: str  # paragraph, figure, table, etc.
    source_info: Dict[str, Any]  # e.g., para_idx, fig_id, table_id
    content: str
    confidence: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def create(cls, paper_id: str, doc_path: str, evidence_type: str, source_info: Dict[str, Any], 
               content: str, confidence: float = 0.7):
        return cls(
            evidence_id=str(uuid.uuid4()),
            paper_id=paper_id,
            doc_path=doc_path,
            evidence_type=evidence_type,
            source_info=source_info,
            content=content,
            confidence=confidence
        )

@dataclass
class Edge:
    edge_id: str
    relationship_type: str
    source_node_id: str
    target_node_id: str
    evidence_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    value: Optional[Any] = None
    unit: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __eq__(self, other):
        if isinstance(other, Edge):
            return self.edge_id == other.edge_id
        return False

    def __hash__(self):
        return hash(self.edge_id)

@dataclass
class KnowledgeGraph:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    evidences: Dict[str, Evidence] = field(default_factory=dict)

    def add_node(self, node: Node) -> str:
        self.nodes[node.node_id] = node
        return node.node_id

    def add_edge(self, edge: Edge) -> str:
        self.edges.append(edge)
        return edge.edge_id

    def add_evidence(self, evidence: Evidence) -> str:
        self.evidences[evidence.evidence_id] = evidence
        return evidence.evidence_id

    def get_node_by_key(self, node_type: str, key: str) -> Optional[Node]:
        for node in self.nodes.values():
            if node.node_type == node_type and node.key == key:
                return node
        return None

    def get_edges_by_node(self, node_id: str) -> List[Edge]:
        result = []
        for edge in self.edges:
            if edge.source_node_id == node_id or edge.target_node_id == node_id:
                result.append(edge)
        return result

    def get_edges_by_relationship(self, relationship_type: str) -> List[Edge]:
        return [edge for edge in self.edges if edge.relationship_type == relationship_type]

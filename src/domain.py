# src/domain.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum
from collections import defaultdict


class ProductType(Enum):
    SOYBEAN = "SOYBEAN"
    SOYBEAN_CAKE = "SOYBEAN_CAKE"
    SOYBEAN_OIL = "SOYBEAN_OIL"


class NodeType(Enum):
    # Supplier nodes
    PRODUCTION = "PRODUCTION"
    # Aggregator nodes
    SILO_LOCAL = "SILO_LOCAL"
    SILO_AGGREGATOR = "SILO_AGGREGATOR"
    HUB = "HUB"
    # Industry
    PROCESSING = "PROCESSING"
    # Multi-modal transhipment nodes
    PORT = "PORT"
    TRAIN = "TRAIN"


@dataclass
class Node:
    id: str
    type: NodeType
    capacity: float = 100.0
    production: float = 0.0       
    inventory_cost: float = 50

    # Ex: {ProductType.SOYBEAN: 1000, ProductType.SOYBEAN_CAKE: 500}
    contract_demands: Dict[ProductType, float] = field(default_factory=lambda: defaultdict(float))

    # NEW: Initial Stock is now a dictionary: {'Soybean': 100, 'Soybean_Cake': 0}
    initial_inventory: Dict[str, float] = field(default_factory=dict)


@dataclass
class Edge:
    source_id: str
    target_id: str
    unit_cost: float
    mode: str = "truck"
    max_capacity: float = 90_000_000 # infinity capacity for all routes
    
    # HYBRID LOGIC: Now specific to a product
    fixed_flow: Optional[float] = None 
    fixed_product: Optional[ProductType] = None # If None, this edge is generic


@dataclass
class FlowConstraint:
    """Representa um contrato comercial ou obrigação logística."""
    source_id: str
    target_id: str
    product: ProductType
    volume: float
    type: str = 'min'  # 'min' (>=), 'max' (<=), 'equal' (==)


@dataclass
class SupplyChainNetwork:
    nodes: Dict[str, 'Node'] = field(default_factory=dict)
    edges: List['Edge'] = field(default_factory=list)
    
    constraints: List[FlowConstraint] = field(default_factory=list)

    def add_node(self, node: 'Node'):
        self.nodes[node.id] = node

    def add_edge(self, edge: 'Edge'):
        self.edges.append(edge)

    def add_constraint(self, source_id: str, target_id: str, product, volume: float, type: str = 'min'):
        """Registra uma obrigação de fluxo entre dois nós."""
        self.constraints.append(
            FlowConstraint(
                source_id=source_id,
                target_id=target_id,
                product=product,
                volume=volume,
                type=type
            )
        )
import pandas as pd
from typing import Dict, Any
from collections import defaultdict
from .domain import Node, NodeType, SupplyChainNetwork, Edge, ProductType

def preparation(network: SupplyChainNetwork, paths: dict, abiove_ref: dict = {}):
    print("Starting Data Pipeline Preparation...")

    # 1. Load Nodes
    ProductionNodes(network, paths['production']).run()
    SiloNodes(network, paths['silos']).run()
    PortNodes(network, paths['ports']).run()
    TrainNodes(network, paths['train']).run()
    ProcessingNodes(network, paths['industrial_capacity']).run()

    # 2. Constraints & Fixed Flows (Can generate missing nodes automatically)
    FixedFlowsConstraints(network, paths['fixed_flows']).run()
    TrainConstraints(network, paths['rail_flows']).run()

    # 3. Create Edges (Explosion Matrix)
    TruckCostMatrix(network, paths['truck_costs']).run()

    print(f"Preparation Complete. Total Nodes: {len(network.nodes)} | Total Edges: {len(network.edges)}")
    return network



class NetworkDataLoader:
    """
    Base class for loading data into the SupplyChainNetwork.
    Mimics the Preprocessor style.
    """
    file_path = None  # To be defined by child classes if they read a file

    def __init__(self, network: SupplyChainNetwork, custom_path: str = None):
        self.network = network
        if custom_path:
            self.file_path = custom_path

    def run(self, *args, **kwargs):
        if self.file_path:
            print(f"[{self.__class__.__name__}] Loading data from {self.file_path}...")
            df = pd.read_csv(self.file_path, delimiter=';')
            return self.process(df, *args, **kwargs)
        else:
            print(f"[{self.__class__.__name__}] Running data processor...")
            return self.process(None, *args, **kwargs)

    def process(self, df: pd.DataFrame, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the process method.")


# =====================================================================
# NODE LOADERS
# =====================================================================
class ProductionNodes(NetworkDataLoader):
    def process(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            node = Node(
                id=str(row['node_id']),
                type=NodeType.PRODUCTION,
                production=float(row['volume']),
                inventory_cost=1_000_000  # Everything should leave the farm
            )
            self.network.add_node(node)
        return df


class SiloNodes(NetworkDataLoader):
    def process(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            silo_type = row.get('facility_type', 'SILO_AGGREGATOR')
            trase_id = str(row['node_id'])
            
            node = Node(
                id=f'{silo_type}_{trase_id}',
                type=NodeType(silo_type),
                capacity=float(row['capacity_tons'])
            )
            self.network.add_node(node)
        return df


class PortNodes(NetworkDataLoader):
    def process(self, df: pd.DataFrame):
        df_ports = df.groupby(by=['target_id', 'product_type'], as_index=False)['volume'].sum()
        df_ports_pivot = df_ports.pivot(index='target_id', columns='product_type', values='volume').fillna(0).reset_index()
        
        for _, row in df_ports_pivot.iterrows():
            vol_cake = float(pd.to_numeric(row.get(ProductType.SOYBEAN_CAKE.value, 0), errors="coerce") or 0)
            vol_oil = float(pd.to_numeric(row.get(ProductType.SOYBEAN_OIL.value, 0), errors="coerce") or 0)
            vol_bean = float(pd.to_numeric(row.get(ProductType.SOYBEANS.value, 0), errors="coerce") or 0)

            node = Node(
                id=f"PORT_{row['target_id']}",
                type=NodeType.PORT,
                capacity=(vol_bean + vol_cake + vol_oil),
                contract_demands={
                    ProductType.SOYBEANS: vol_bean,
                    ProductType.SOYBEAN_CAKE: vol_cake,
                    ProductType.SOYBEAN_OIL: vol_oil
                }
            )
            self.network.add_node(node)
        return df_ports_pivot


class TrainNodes(NetworkDataLoader):
    def process(self, df: pd.DataFrame):
        df_departure = df.groupby(['origin_id', 'product_type'])['volume'].sum().reset_index()
        df_arrival = df.groupby(['dest_id', 'product_type'])['volume'].sum().reset_index()

        df_dep_pivot = df_departure.pivot_table(index='origin_id', columns='product_type', values='volume', fill_value=0).reset_index()
        df_arr_pivot = df_arrival.pivot_table(index='dest_id', columns='product_type', values='volume', fill_value=0).reset_index()
        df_dep_pivot.columns.name = None
        df_arr_pivot.columns.name = None

        self._create_train_nodes(df_arr_pivot, 'dest_id')
        self._create_train_nodes(df_dep_pivot, 'origin_id')
        return df

    def _create_train_nodes(self, df_pivot, id_col):
        for _, row in df_pivot.iterrows():
            node = Node(
                id=str(row[id_col]),
                type=NodeType.TRAIN,
                capacity=float(row.get('SOYBEANS', 0) + row.get('SOYBEAN_CAKE', 0)),
                contract_demands={
                    ProductType.SOYBEANS: float(row.get('SOYBEANS', 0)),
                    ProductType.SOYBEAN_CAKE: float(row.get('SOYBEAN_CAKE', 0)),
                }
            )
            self.network.add_node(node)


class ProcessingNodes(NetworkDataLoader):
    def process(self, df: pd.DataFrame):
        NATIONAL_VOL_CAKE = 42_292_000
        
        total_capacity = df["capacity_tons"].sum() * 1.031
        df["vol_processed_beans_for_cake"] = (df["capacity_tons"] * NATIONAL_VOL_CAKE / total_capacity * 1.031)
        df["vol_processed_beans_for_oil"] = df["vol_processed_beans_for_cake"] * (0.19 / 0.78)
        
        for _, row in df.iterrows():
            node = Node(
                id=str(row['node_id']),
                type=NodeType.PROCESSING,
                capacity=float(row['capacity_tons']),
                inventory_cost=3.0,
                contract_demands={
                    ProductType.SOYBEAN_CAKE: float(row.get('vol_processed_beans_for_cake', 0)),
                    ProductType.SOYBEAN_OIL: float(row.get('vol_processed_beans_for_oil', 0))
                }
            )
            self.network.add_node(node)
        return df


# =====================================================================
# CONSTRAINTS & EDGES LOADERS
# =====================================================================
class MissingNodesFixer(NetworkDataLoader):
    """Helper processor used by FixedFlows to auto-generate missing nodes."""
    def process(self, df_grouped: pd.DataFrame):
        for _, row in df_grouped.iterrows():
            source_id = row['source_id']
            volume = float(row['volume'])

            try:
                node_type = NodeType(source_id.split('_')[0])
            except ValueError:
                continue

            if source_id not in self.network.nodes:
                if source_id.startswith('PRODUCTION'):
                    source_id = source_id.replace('PRODUCTION', 'HUB')
                    node_type = NodeType.HUB
                    capacity = volume
                elif node_type == NodeType.PROCESSING:
                    capacity = (volume / 0.78) * 1.5
                    print(f"⚠️ Created Ghost Industry: {source_id} (Cap: {capacity:.0f}t)")
                else:
                    capacity = volume

                self.network.add_node(Node(
                    id=source_id, 
                    type=node_type, 
                    capacity=capacity,
                    inventory_cost=3.0 if node_type == NodeType.PROCESSING else 5.0
                ))
        return None


class FixedFlowsConstraints(NetworkDataLoader):
    def process(self, df: pd.DataFrame):
        df = df[df['branch'].str.startswith(('1.', '2.', '3.1', '3.2.1'))].copy()

        # Prefix rules
        df.loc[df['branch'].str.startswith('1.'), 'source_id'] = 'PRODUCTION_' + df['source_id'].astype(str)
        df.loc[df['branch'].str.startswith('2.'), 'source_id'] = 'HUB_' + df['source_id'].astype(str)
        df.loc[df['branch'].str.startswith(('3.1', '3.2.1')), 'source_id'] = 'PROCESSING_' + df['source_id'].astype(str)

        df['target_id'] = 'PORT_' + df['target_id'].fillna('').astype(str)
        df['product_type'] = df['product_type'].str.replace(' ', '_')

        df_grouped = df.groupby(['product_type', 'source_id', 'target_id'], as_index=False)['volume'].sum()

        # Fix missing nodes before applying constraints
        MissingNodesFixer(self.network).process(df_grouped)

        for _, row in df_grouped.iterrows():
            try:
                product_enum = ProductType(str(row['product_type']).strip())
            except ValueError:
                continue
            
            source_id = str(row['source_id']).replace('PRODUCTION', 'HUB') if str(row['source_id']).startswith('PRODUCTION') and str(row['source_id']) not in self.network.nodes else str(row['source_id'])
            target_id = str(row['target_id'])

            if not any(e.source_id == source_id and e.target_id == target_id for e in self.network.edges):
                self.network.add_edge(Edge(source_id=source_id, target_id=target_id, mode='truck'))

            self.network.add_constraint(
                source_id=source_id,
                target_id=target_id,
                product=product_enum,
                volume=float(row['volume']),
                type='min'
            )
        return df


class TrainConstraints(NetworkDataLoader):
    def process(self, df: pd.DataFrame):
        df_grouped = df.groupby(['dest_id', 'origin_id', 'product_type'])['volume'].sum().reset_index()
        
        for _, row in df_grouped.iterrows():
            origin, dest = str(row['origin_id']), str(row['dest_id'])
            if origin in self.network.nodes and dest in self.network.nodes:
                self.network.add_edge(
                    Edge(
                        source_id=origin, 
                        target_id=dest, 
                        mode='train', 
                        unit_cost=0,
                        fixed_flow=float(row['volume']),
                        fixed_product=ProductType(str(row['product_type']).strip().upper())
                    )
                )
        return df


class TruckCostMatrix(NetworkDataLoader):
    def process(self, df: pd.DataFrame):
        df = df[df['cost'] < 50].copy()
        df['cost'] = df['cost'].mul(3600)
        df['origin'] = df['origin'].str.replace('-', '_')
        df['destination'] = df['destination'].str.replace('-', '_')

        # Map nodes using geo key
        nodes_by_loc = defaultdict(list)
        for node in self.network.nodes.values():
            start_index = node.id.find("BR_")
            geo_key = node.id[start_index:] if start_index != -1 else node.id
            nodes_by_loc[geo_key].append(node)

        count, ignored = 0, 0
        for _, row in df.iterrows():
            orig_key, dest_key = str(row['origin']).strip(), str(row['destination']).strip()
            
            if orig_key not in nodes_by_loc or dest_key not in nodes_by_loc:
                continue

            for src_node in nodes_by_loc[orig_key]:
                for dst_node in nodes_by_loc[dest_key]:
                    if src_node.id == dst_node.id or not self.is_valid_route(src_node, dst_node, float(row['cost'])):
                        ignored += 1
                        continue
                    
                    self.network.add_edge(Edge(
                        source_id=src_node.id, target_id=dst_node.id, 
                        mode='rail' if src_node.type == NodeType.TRAIN else 'truck',
                        unit_cost=float(row['cost'])
                    ))
                    count += 1
        print(f"[{self.__class__.__name__}] Created {count} valid edges. Ignored {ignored}.")
        return df

    @staticmethod
    def is_valid_route(src: Node, dst: Node, cost: float) -> bool:
        """Business validation rules for edges."""
        if dst.type == NodeType.PORT and cost < 4 * 3600: return True
        if str(src.id).split('_')[1] == str(dst.id).split('_')[1] and dst.type == NodeType.PORT: return True
        
        if src.type in [NodeType.SILO_AGGREGATOR, NodeType.SILO_LOCAL]:
            return dst.type in [NodeType.SILO_AGGREGATOR, NodeType.PROCESSING, NodeType.HUB, NodeType.TRAIN]
        if src.type == NodeType.HUB:
            return dst.type in [NodeType.HUB, NodeType.PORT, NodeType.TRAIN]
        if src.type == NodeType.PROCESSING:
            return dst.type in [NodeType.HUB, NodeType.TRAIN]
        if src.type == NodeType.PRODUCTION:
            return dst.type in [NodeType.PROCESSING, NodeType.HUB, NodeType.TRAIN, NodeType.SILO_LOCAL, NodeType.SILO_AGGREGATOR]
        if src.type == NodeType.TRAIN:
            if dst.type == NodeType.PORT and cost < 5 * 3600: return True
            return dst.type in [NodeType.TRAIN, NodeType.PROCESSING, NodeType.HUB]
        return False


class InitialStocksDistribution(NetworkDataLoader):
    """Distributes initial carry-over stock into silos based on capacity ratio. No CSV required."""
    def process(self, df: Any, abiove_reference: Dict):
        total_cap = sum(n.capacity for n in self.network.nodes.values() if n.type in [NodeType.SILO_AGGREGATOR, NodeType.SILO_LOCAL, NodeType.HUB])
        if total_cap == 0: return

        beans_stock = abiove_reference[ProductType.SOYBEANS.value]['Initial stock']
        cake_stock = abiove_reference[ProductType.SOYBEAN_CAKE.value]['Initial stock']
        oil_stock = abiove_reference[ProductType.SOYBEAN_OIL.value]['Initial stock']

        for node in self.network.nodes.values():
            ratio = node.capacity / total_cap
            if node.type in [NodeType.SILO_AGGREGATOR, NodeType.SILO_LOCAL, NodeType.HUB]:
                node.initial_inventory = {ProductType.SOYBEANS.value: beans_stock * ratio}
            elif node.type == NodeType.PROCESSING:
                node.initial_inventory = {
                    ProductType.SOYBEAN_CAKE.value: cake_stock * ratio,
                    ProductType.SOYBEAN_OIL.value: oil_stock * ratio
                }
        return None
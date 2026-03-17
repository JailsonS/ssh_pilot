import pulp
from typing import List, Iterable
from .domain import ProductType, Edge
from collections import defaultdict

class SupplyChainSolver:
    def __init__(self, name="Soy_Optimization"):
        self.prob = pulp.LpProblem(name, pulp.LpMinimize)
        
        self.flow_vars = {}      # k: (src, dst, mode, product)
        self.storage_vars = {}   # k: (node_id, product)
        self.slack_vars = {}     # k: (node_id, slack_type, product)
        
        # List of accumulated parts of the objective function
        self._objective_terms = []

        self._inbound_map = None
        self._outbound_map = None

    # ==========================================
    # 1. CREATE VARIABLES 
    # ==========================================

    def add_flow_variables(self, edges: Iterable[Edge], products: List[ProductType]):
        """Create variables respecting edge strict product constraints to save memory."""
        for edge in edges:
            
            allowed_products = [edge.fixed_product] if edge.fixed_product else products
            
            for prod in allowed_products:
                var_name = f"Flow_{edge.source_id}_{edge.target_id}_{edge.mode}_{prod.value}"
                
                lb, ub = 0, edge.max_capacity
                if edge.fixed_flow is not None and edge.fixed_product == prod:
                    lb = ub = edge.fixed_flow
                
                var = pulp.LpVariable(var_name, lowBound=lb, upBound=ub, cat='Continuous')
                self.flow_vars[(edge.source_id, edge.target_id, edge.mode, prod)] = var


    def add_storage_variables(self, node_ids: List[str], products: List[ProductType]):
        """Create stock variables."""
        for n_id in node_ids:
            for prod in products:
                var_name = f"Stock_{n_id}_{prod.value}"
                var = pulp.LpVariable(var_name, lowBound=0, cat='Continuous')
                self.storage_vars[(n_id, prod)] = var


    # ==========================================
    # 2. CREATE SLACKS 
    # ==========================================

    def add_supply_slacks(self, node_ids: List[str], penalty_cost: float, product: ProductType = None):
        """Injects 'shortage' variables (dummy supply) into specific nodes and adds the cost."""
        prod_suffix = f"_{product.value}" if product else ""
        
        for n_id in node_ids:
            var_name = f"SlackSupply_{n_id}{prod_suffix}"
            var = pulp.LpVariable(var_name, lowBound=0, cat='Continuous')
            self.slack_vars[(n_id, 'supply', product)] = var
            
            # Automatically injects the cost of this slack into the objective function
            self.add_to_objective(var * penalty_cost)


    def add_sink_slacks(self, node_ids: List[str], penalty_cost: float, product: ProductType = None):
        """Injects 'surplus' variables (dummy sink/waste) into specific nodes."""
        prod_suffix = f"_{product.value}" if product else ""
        
        for n_id in node_ids:
            var_name = f"SlackSink_{n_id}{prod_suffix}"
            var = pulp.LpVariable(var_name, lowBound=0, cat='Continuous')
            self.slack_vars[(n_id, 'sink', product)] = var
            self.add_to_objective(var * penalty_cost)


    # ==========================================
    # INDEXING UTILITIES (Performance Magic)
    # ==========================================
    def prepare_flow_indexes(self):
        """Scans all registered flow variables and creates O(1) maps for fast access."""
        print("Preparing high-performance flow indexes...")
        self._inbound_map = defaultdict(lambda: defaultdict(list))
        self._outbound_map = defaultdict(lambda: defaultdict(list))

        for (src, dst, mode, product), var in self.flow_vars.items():
            self._outbound_map[src][product].append(var)
            self._inbound_map[dst][product].append(var)


    def get_inbound_flows(self, node_id: str, product: ProductType) -> List[pulp.LpVariable]:
        """Returns the list of variables ARRIVING at the node for the given product."""
        if self._inbound_map is None:
            raise RuntimeError("You must call prepare_flow_indexes() before fetching flows.")
        return self._inbound_map[node_id][product]


    def get_outbound_flows(self, node_id: str, product: ProductType) -> List[pulp.LpVariable]:
        """Returns the list of variables LEAVING the node for the given product."""
        if self._outbound_map is None:
            raise RuntimeError("You must call prepare_flow_indexes() before fetching flows.")
        return self._outbound_map[node_id][product]


    def get_storage_var(self, node_id: str, product: ProductType) -> pulp.LpVariable:
        """Fetches the storage variable with a fallback to zero if it doesn't exist."""
        return self.storage_vars.get((node_id, product), 0)


    def get_slack_var(self, node_id: str, slack_type: str, product: ProductType = None):
        """Returns the supply/sink slack, or zero if it hasn't been injected in this node."""
        return self.slack_vars.get((node_id, slack_type, product), 0)

    # ==========================================
    # CONSTRAINTS AND OBJECTIVE
    # ==========================================
    def add_custom_constraint(self, constraint_expression, name: str):
        self.prob += constraint_expression, name


    def add_to_objective(self, cost_expression):
        self._objective_terms.append(cost_expression)


    def solve(self, msg=1, threads=4):
        self.prob += pulp.lpSum(self._objective_terms), "Total_Cost_Objective"
        status = self.prob.solve(pulp.PULP_CBC_CMD(msg=msg, threads=threads))
        return status
    

# src/optimizer.py
import pulp
from .domain import SupplyChainNetwork, NodeType, ProductType
from collections import defaultdict
import pandas as pd

class SoyChainSolver:
    
    def __init__(self, network: SupplyChainNetwork):
        self.network = network
        self.prob = pulp.LpProblem("Soy_MultiCommodity_Optimization", pulp.LpMinimize)
        
        # Keys: (src, dst, mode, product_name)
        self.flow_vars = {} 
        # Keys: (node_id, product_name)
        self.storage_vars = {} 
        # Keys: (node_id, product_name)
        self.export_vars = {} 
        # Keys: (node_id, product_name)
        self.domestic_vars = {} 


        # Keys: node_id
        self.dummy_supply_vars = {}
        self.dummy_sink_vars = {}
        
        self.processing_slacks = {}
        
        # Conversion Factors (Crushing Yields)
        self.YIELD_CAKE = 0.78
        self.YIELD_OIL = 0.19

        self.P_CONTRACT = 1_000_000.0   # Sufficient to be 100x higher than freight
        self.P_DUMMY = 500_000.0        # The last resort
        self.P_MAGIC_GEN = 500_000.0    # Magic generation in industry (avoid as much as possible)
        self.EXPORT_REWARD = -100.0     # Incentive for the port to "pull" the cargo
        self.DOMESTIC_REWARD = -50.0
        

    def build_model(self):
        print("Building Multi-Commodity LP Model...")
        
        self._create_variables()
        self._add_slack_variables() 
        
        self._add_node_balance_constraints()
        self._add_fixed_flow_constraints()
        
        self._set_objective()


    def _create_variables(self):
        # 1. Transport Variables
        for edge in self.network.edges:
            # Determine allowed products for this edge
            if edge.fixed_product:
                allowed_products = [edge.fixed_product] # e.g. Contract for OIL only
            else:
                allowed_products = [p for p in ProductType] # Trucks can carry anything

            for prod in allowed_products:
                var_name = f"Flow_{edge.source_id}_{edge.target_id}_{edge.mode}_{prod.value}"
                
                # Check Fixed Flow (Decision Tree)
                lb, ub = 0, edge.max_capacity
                if edge.fixed_flow is not None and edge.fixed_product == prod:
                    lb = ub = edge.fixed_flow
                
                self.flow_vars[(edge.source_id, edge.target_id, edge.mode, prod)] = pulp.LpVariable(
                    var_name, lowBound=lb, upBound=ub, cat='Continuous'
                )

        # 2. Storage Variables (Per Product)
        for node_id, node in self.network.nodes.items():
            
            # Define what can be stocked based on Node Type
            allowed_stock = []
            
            if node.type in [NodeType.SILO_AGGREGATOR, NodeType.SILO_LOCAL, NodeType.HUB]:
                allowed_stock = [ProductType.SOYBEAN] # Silos only store beans
            
            elif node.type in [NodeType.PROCESSING, NodeType.PORT]:
                allowed_stock = [p for p in ProductType] # Factory/Port stores everything
            
            # Create variables only for allowed products
            for prod in allowed_stock:
                var_name = f"Stock_{node_id}_{prod.value}"
                self.storage_vars[(node_id, prod)] = pulp.LpVariable(
                    var_name, lowBound=0, cat='Continuous'
                )

        # 3. Waste Variables (Mostly for industry)
        self.waste_vars = {}
        for node_id, node in self.network.nodes.items():
            if node.type == NodeType.PROCESSING:
                for prod in [ProductType.SOYBEAN_CAKE, ProductType.SOYBEAN_OIL]:
                    self.waste_vars[(node_id, prod)] = pulp.LpVariable(
                        f"Waste_{node_id}_{prod.value}", 
                        lowBound=0
                    )

        self.domestic_vars = {}
        for node_id, node in self.network.nodes.items():
            if node.type == NodeType.PROCESSING:
                for prod in [ProductType.SOYBEAN_CAKE, ProductType.SOYBEAN_OIL]:
                    self.domestic_vars[(node_id, prod)] = pulp.LpVariable(
                        f"Domestic_{node_id}_{prod.value}", 
                        lowBound=0,
                        cat='Continuous'
                    )


    def _add_slack_variables(self):
        """Creates 'Magical' Dummy Nodes to prevent Infeasibility."""
        print("Adding Dummy/Slack nodes...")
        
        # 1. Shortage Slacks (Dummy Supply) - Kept the same
        for node_id, node in self.network.nodes.items():
            if node.type in [NodeType.PROCESSING]:
                var_name = f"Slack_Shortage_{node_id}"
                self.dummy_supply_vars[node_id] = pulp.LpVariable(
                    var_name, lowBound=0, cat='Continuous'
                )

        # 2. Surplus Slacks (Dummy Sink)
        for node_id, node in self.network.nodes.items():
            # ADDED NodeType.PRODUCTION HERE
            if node.type in [NodeType.PORT, NodeType.SILO_AGGREGATOR, NodeType.PRODUCTION]:
                var_name = f"Slack_Surplus_{node_id}"
                self.dummy_sink_vars[node_id] = pulp.LpVariable(
                    var_name, lowBound=0, cat='Continuous'
                )


    def _set_objective(self):
        print("Setting Global Objective...")
        
        edge_cost_map = {(e.source_id, e.target_id, e.mode): e.unit_cost for e in self.network.edges}
        node_inv_costs = {n_id: n.inventory_cost for n_id, n in self.network.nodes.items()}

        # 1. Transport & Storage
        transport_costs = (v * edge_cost_map.get((s, d, m), 0.0) for (s, d, m, p), v in self.flow_vars.items())
        storage_costs = (v * node_inv_costs[n_id] for (n_id, p), v in self.storage_vars.items())

        # 2. Slacks (Using the new calibrated constants)
        dummy_costs = (v * self.P_DUMMY for v in list(self.dummy_supply_vars.values()) + list(self.dummy_sink_vars.values()))
        contract_costs = (v * self.P_CONTRACT for v in self.contract_slacks.values())
        
        # Processing Slacks (adjusted to be read from the correct dictionary)
        magic_processing_costs = []
        for cake_s, oil_s in self.processing_slacks.values():
            magic_processing_costs.append(cake_s * self.P_MAGIC_GEN)
            magic_processing_costs.append(oil_s * self.P_MAGIC_GEN)

        # 3. Rewards (Export Incentive)
        # We verify if export_vars exists to avoid errors
        rewards = (v * self.EXPORT_REWARD for v in getattr(self, 'export_vars', {}).values())
        domestic_rewards = (v * self.DOMESTIC_REWARD for v in getattr(self, 'domestic_vars', {}).values())

        fixed_flow_penalties = (v * 2_000_000.0 for v in self.fixed_flow_slacks.values())
        
        self.prob += (
            pulp.lpSum(transport_costs) + 
            pulp.lpSum(storage_costs) + 
            pulp.lpSum(dummy_costs) + 
            pulp.lpSum(contract_costs) +
            pulp.lpSum(fixed_flow_penalties) +
            pulp.lpSum(magic_processing_costs) +
            pulp.lpSum(rewards) + 
            pulp.lpSum(domestic_rewards)
        )


    def _add_node_balance_constraints(self):
        print(f"Indexing flows for {len(self.flow_vars)} variables...")
        
        # 1. Indexing
        inbound_map = defaultdict(lambda: defaultdict(list))
        outbound_map = defaultdict(lambda: defaultdict(list))

        for (supply_node, dest_node, mode, product), var in self.flow_vars.items():
            outbound_map[supply_node][product].append(var)
            inbound_map[dest_node][product].append(var)

        # Dictionary to store Contract Slacks (for auditing later)
        self.contract_slacks = {}

        print("Adding Multi-Commodity Balance Constraints...")
        
        for node_id, node in self.network.nodes.items():
                        
            # --- CASE 1: PRODUCTION (FARM) ---
            if node.type == NodeType.PRODUCTION:
                beans_out = pulp.lpSum(outbound_map[node_id][ProductType.SOYBEAN])
                
                stock_final = self.storage_vars.get((node_id, ProductType.SOYBEAN), 0)
                
                # Error Slack (only if the model cannot stock nor transport)
                slack_error = self.dummy_sink_vars.get(node_id, 0)
                
                # NEW EQUATION: What was harvested has 3 destinations:
                # 1. Use truck/train (Cost: Freight)
                # 2. Stock cost (Cost: 1000 - YOUR NEW COST)
                # 3. Use Slack (Cost: 1,000,000 - MAXIMUM PENALTY)
                self.prob += (beans_out + stock_final + slack_error) == node.production, f"Supply_{node_id}"

            # --- CASE 2: INDUSTRY/PROCESSING ---
            elif node.type == NodeType.PROCESSING:
                # Grain Input (Physical + Spot)
                beans_in = pulp.lpSum(inbound_map[node_id][ProductType.SOYBEAN])
                slack_in_beans = self.dummy_supply_vars.get(node_id, 0) 
                
                # Check beans out to block it
                beans_out = pulp.lpSum(outbound_map[node_id][ProductType.SOYBEAN])
                
                # Physical outputs
                cake_out = pulp.lpSum(outbound_map[node_id][ProductType.SOYBEAN_CAKE])
                oil_out = pulp.lpSum(outbound_map[node_id][ProductType.SOYBEAN_OIL])
                
                # Stocks
                stock_cake = self.storage_vars.get((node_id, ProductType.SOYBEAN_CAKE), 0)
                stock_oil = self.storage_vars.get((node_id, ProductType.SOYBEAN_OIL), 0)
                stock_beans = self.storage_vars.get((node_id, ProductType.SOYBEAN), 0)

                # Domestic Demands
                domestic_cake = self.domestic_vars.get((node_id, ProductType.SOYBEAN_CAKE), 0)
                domestic_oil = self.domestic_vars.get((node_id, ProductType.SOYBEAN_OIL), 0)
                
                # Generation Slacks (Magic Generation) - Create unique names with ID
                slack_make_cake = pulp.LpVariable(f"Magic_Gen_Cake_{node_id}", lowBound=0)
                slack_make_oil = pulp.LpVariable(f"Magic_Gen_Oil_{node_id}", lowBound=0)
                self.processing_slacks[node_id] = (slack_make_cake, slack_make_oil)

                # ==========================================================
                # The Carousel lock: Only the soybeans that were NOT shipped out as grain
                # and did NOT remain in inventory are considered "Crushed"
                # ==========================================================
                crushed_beans = beans_in + slack_in_beans - beans_out - stock_beans

                # Yield restrictions
                self.prob += (crushed_beans * self.YIELD_CAKE) + slack_make_cake == (
                    cake_out + stock_cake + domestic_cake + self.waste_vars.get((node_id, ProductType.SOYBEAN_CAKE), 0)
                ), f"Yield_Cake_{node_id}"

                self.prob += (crushed_beans * self.YIELD_OIL) + slack_make_oil == (
                    oil_out + stock_oil + domestic_oil + self.waste_vars.get((node_id, ProductType.SOYBEAN_OIL), 0)
                ), f"Yield_Oil_{node_id}"
                
                # Avoid negative crushing
                self.prob += crushed_beans >= 0, f"Positive_Crush_{node_id}"
                            
            # --- CASE 3: SILOS/PORTS ---
            else:
                for product in ProductType:
                    flow_in = pulp.lpSum(inbound_map[node_id][product])
                    flow_out = pulp.lpSum(outbound_map[node_id][product])
                    
                    stock_final = self.storage_vars.get((node_id, product), 0)
                    initial_inv = node.initial_inventory.get(product.value, 0)
                    
                    # Port logic
                    if node.type == NodeType.PORT:
                        # Port is a SINK what goes in disappears to external market
                        target_volume = node.contract_demands.get(product, 0)
#
                        exported_var = pulp.LpVariable(
                            f"Exported_{node_id}_{product.value}", 
                            lowBound=0, 
                            upBound=target_volume 
                        )
                       
                        # 
                        self.export_vars = getattr(self, 'export_vars', {})
                        self.export_vars[(node_id, product)] = exported_var
                       
                        # Restriction: Everything that comes in goes out as "Exported" (or stays in stock if we want to store it)
                        self.prob += (flow_in + initial_inv) == (flow_out + stock_final + exported_var)
                            
                    else:
                        # Silos/Hubs (Standard behavior)
                        slack_out = self.dummy_sink_vars.get(node_id, 0)
                        self.prob += (flow_in + initial_inv) == (flow_out + stock_final + slack_out)

                        if node.capacity and node.capacity > 0:
                            self.prob += stock_final <= node.capacity, f"Max_Stock_{node_id}_{product.value}"

                        if node.type == NodeType.TRAIN:
                            # Railway Hubs: Giant throughput, zero retained stock.
                            self.prob += flow_in <= (node.capacity * 25.0) # Turnover 25x
                            self.prob += stock_final == 0 # Forbidden to store from one year to the next

                        elif node.type == NodeType.HUB:
                            # Road or Waterway Hubs (Transshipment)
                            self.prob += flow_in <= (node.capacity * 15.0) # Turnover 15x
                            self.prob += stock_final <= (node.capacity * 0.1) # Max 10% remains for the end of the year

                        elif node.type in [NodeType.SILO_AGGREGATOR, NodeType.SILO_LOCAL]:
                            # The Interior Lungs (Where the real stock stays)
                            self.prob += flow_in <= (node.capacity * 3.0) # Turnover 3x (Soybean harvest + Corn off-season)
                            # No need to force stock_final = 0. Let the solver decide to store here


            # CASE 4: CONTRACTS - DECISION TREE OUTPUTS 
            for product, required_amount in node.contract_demands.items():
                if required_amount > 0.001:
                    flow_in_contract = pulp.lpSum(inbound_map[node_id][product])
                    
                    # Unique name for the contract slack to avoid overlap
                    slack_name = f"Missed_Contract_{node_id}_{product.value}".replace("-", "_")
                    slack_contract = pulp.LpVariable(slack_name, lowBound=0)
                    self.contract_slacks[slack_name] = slack_contract
                    
                    self.prob += (flow_in_contract + slack_contract) >= required_amount, f"Contract_Req_{node_id}_{product.value}"
    
    
    def _add_fixed_flow_constraints(self):
        print("Applying Fixed Flow Constraints (Contracts) with SLACK protection...")
        
        if not hasattr(self.network, 'constraints'):
            return

        flow_vars_by_link = defaultdict(list)
        for (src, dst, mode, prod), var in self.flow_vars.items():
            flow_vars_by_link[(src, dst, prod)].append(var)

        count = 0
        self.fixed_flow_slacks = {} #

        for i, constr in enumerate(self.network.constraints):
            relevant_vars = flow_vars_by_link.get((constr.source_id, constr.target_id, constr.product), [])
            
            # 1. Create variable for "Contract Breach"
            safe_id = f"{constr.source_id}_{constr.target_id}".replace("-", "_").replace(".", "_")
            slack_name = f"Missed_FixedFlow_{i}_{safe_id}"
            
            slack_var = pulp.LpVariable(slack_name, lowBound=0)
            self.fixed_flow_slacks[slack_name] = slack_var # Save to penalize later

            constr_name = f"FixedFlow_{i}_{safe_id}_{constr.product.value}"

            # 2. Add Slack in the equation: (Real Flow + What was missing) == Contracted Volume
            if constr.type == 'min':
                self.prob += (pulp.lpSum(relevant_vars) + slack_var) >= constr.volume, constr_name
            elif constr.type == 'equal':
                self.prob += (pulp.lpSum(relevant_vars) + slack_var) == constr.volume, constr_name
            elif constr.type == 'max':
                self.prob += (pulp.lpSum(relevant_vars) + slack_var) <= constr.volume, constr_name
            
            count += 1
            
        print(f"   -> Applied {count} fixed flow constraints (Safe Mode).")


    def solve(self):
        print("\nStarting solver (this may take a while for 5.8M edges)...")
        
        # msg=1 activates CBC log
        # threads=N allows using more cores of your processor
        status = self.prob.solve(pulp.PULP_CBC_CMD(msg=1, threads=4)) 
        
        print(f"Status: {pulp.LpStatus[status]}")
        return status
    
# 
# Identify same name variables, merging costs
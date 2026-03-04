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

        self.P_CONTRACT = 2_000_000.0   # Sufficient to be 100x higher than freight
        self.P_DUMMY = 400_000.0        # The last resort
        self.P_MAGIC_GEN = 400_000.0    # Magic generation in industry (avoid as much as possible)
        self.EXPORT_REWARD = -1_000_000.0     # Incentive for the port to "pull" the cargo
        self.DOMESTIC_REWARD = -100_000.0
        

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
                allowed_stock = [ProductType.SOYBEANS] # Silos only store beans

            elif node.type == NodeType.PROCESSING:
                # Indústria guarda grão para esmagar e guarda farelo/óleo para vender
                allowed_stock = [ProductType.SOYBEANS, ProductType.SOYBEAN_CAKE, ProductType.SOYBEAN_OIL]
                
            elif node.type == NodeType.PORT:
                # Portos guardam tudo aguardando o navio
                allowed_stock = [ProductType.SOYBEANS, ProductType.SOYBEAN_CAKE, ProductType.SOYBEAN_OIL]
            
            # Create variables only for allowed products
            for prod in allowed_stock:
                var_name = f"Stock_{node_id}_{prod.value}"
                self.storage_vars[(node_id, prod)] = pulp.LpVariable(
                    var_name, lowBound=0, cat='Continuous'
                )

        # 3. Domestic
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

        # 3. Waste Variables (Mostly for industry)
        self.waste_vars = {}
        for node_id, node in self.network.nodes.items():
            if node.type == NodeType.PROCESSING:
                for prod in [ProductType.SOYBEAN_CAKE, ProductType.SOYBEAN_OIL]:
                    self.waste_vars[(node_id, prod)] = pulp.LpVariable(
                        f"Waste_{node_id}_{prod.value}", 
                        lowBound=0
                    )


    def _set_objective(self):
        print("Setting Global Objective...")
        
        edge_cost_map = {(e.source_id, e.target_id, e.mode): e.unit_cost for e in self.network.edges}
        node_inv_costs = {n_id: n.inventory_cost for n_id, n in self.network.nodes.items()}

        # 1. Transport Costs
        transport_costs = (v * edge_cost_map.get((s, d, m), 0.0) for (s, d, m, p), v in self.flow_vars.items())

        # 2. Storage Costs
        storage_costs = (v * node_inv_costs[n_id] for (n_id, p), v in self.storage_vars.items())

        # 3. Slacks (Using the new calibrated constants)
        dummy_costs = (
            v * self.P_DUMMY 
            for v in 
                list(self.dummy_supply_vars.values()) + 
                list(self.dummy_sink_vars.values()) +
                list(self.waste_vars.values()) 
        )
        
        # 4. Contract Costs
        contract_costs = (v * self.P_CONTRACT for v in self.contract_slacks.values())
        
        # 5. Processing Slacks (adjusted to be read from the correct dictionary)
        magic_processing_costs = (
            slack * self.P_MAGIC_GEN
            for cake_s, oil_s in self.processing_slacks.values()
            for slack in (cake_s, oil_s)
        )

        # 3. Rewards (Export Incentive)
        rewards = (v * self.EXPORT_REWARD for v in self.export_vars.values())
        domestic_rewards = (v * self.DOMESTIC_REWARD for v in self.domestic_vars.values())

        fixed_flow_penalties = (v * self.P_CONTRACT for v in self.fixed_flow_slacks.values())
        
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
                
                beans_out = pulp.lpSum(outbound_map[node_id][ProductType.SOYBEANS])
                cake_out = pulp.lpSum(outbound_map[node_id][ProductType.SOYBEAN_CAKE])
                oil_out = pulp.lpSum(outbound_map[node_id][ProductType.SOYBEAN_OIL])
                
                self.prob += beans_out == node.production, f"Supply_{node_id}"
                self.prob += (cake_out + oil_out) == 0, f"No_Magic_Gen_at_{node_id}" # Don't allow it!


            # --- CASE 2: INDUSTRY/PROCESSING ---
            elif node.type == NodeType.PROCESSING:
                
                # 1. Recuperar Estoques Iniciais (OFERTA FÍSICA)
                init_beans = node.initial_inventory.get(ProductType.SOYBEANS.value, 0)
                init_cake = node.initial_inventory.get(ProductType.SOYBEAN_CAKE.value, 0)
                init_oil = node.initial_inventory.get(ProductType.SOYBEAN_OIL.value, 0)

                # 2. Entradas e Saídas
                beans_in = pulp.lpSum(inbound_map[node_id][ProductType.SOYBEANS])
                beans_out = pulp.lpSum(outbound_map[node_id][ProductType.SOYBEANS])
                slack_in_beans = self.dummy_supply_vars.get(node_id, 0) 
                
                cake_out = pulp.lpSum(outbound_map[node_id][ProductType.SOYBEAN_CAKE])
                oil_out = pulp.lpSum(outbound_map[node_id][ProductType.SOYBEAN_OIL])
                
                # 3. Variáveis de Estoque Final
                stock_cake = self.storage_vars.get((node_id, ProductType.SOYBEAN_CAKE), 0)
                stock_oil = self.storage_vars.get((node_id, ProductType.SOYBEAN_OIL), 0)
                stock_beans = self.storage_vars.get((node_id, ProductType.SOYBEANS), 0)

                # 4. Variáveis de Mercado Doméstico (O "ralo" das sobras)
                domestic_cake = self.domestic_vars.get((node_id, ProductType.SOYBEAN_CAKE), 0)
                domestic_oil = self.domestic_vars.get((node_id, ProductType.SOYBEAN_OIL), 0)
                
                # 5. Variáveis de Folga (Magic Generation)
                slack_make_cake = pulp.LpVariable(f"Magic_Gen_Cake_{node_id}", lowBound=0)
                slack_make_oil = pulp.LpVariable(f"Magic_Gen_Oil_{node_id}", lowBound=0)
                self.processing_slacks[node_id] = (slack_make_cake, slack_make_oil)

                # ==========================================================
                # O CARROSSEL: Grãos Entrantes + Estoque Inicial - Estoque Final
                # ==========================================================
                crushed_beans = beans_in + slack_in_beans + init_beans - stock_beans 

                # Restrições de Rendimento (Oferta = Demanda)
                # Oferta: (Esmagamento * Rendimento) + Geração Mágica + ESTOQUE INICIAL
                # Demanda: Saída (Export/Hubs) + Estoque Final + Mercado Doméstico + Lixo
                
                self.prob += (crushed_beans * self.YIELD_CAKE) + slack_make_cake + init_cake == (
                    cake_out + stock_cake + domestic_cake + self.waste_vars.get((node_id, ProductType.SOYBEAN_CAKE), 0)
                ), f"Yield_Cake_{node_id}"

                self.prob += (crushed_beans * self.YIELD_OIL) + slack_make_oil + init_oil == (
                    oil_out + stock_oil + domestic_oil + self.waste_vars.get((node_id, ProductType.SOYBEAN_OIL), 0)
                ), f"Yield_Oil_{node_id}"
                
                # Regras de Negócio
                self.prob += crushed_beans >= 0, f"Positive_Crush_{node_id}"
                # self.prob += crushed_beans <= node.capacity * 1.2, f"Max_Crush_Capacity_{node_id}"
                self.prob += beans_out == 0, f"No_Beans_Out_{node_id}"

                # ==========================================================
                # Força a Indústria a produzir e escoar a cota definida
                # ==========================================================
                for prod in [ProductType.SOYBEAN_CAKE, ProductType.SOYBEAN_OIL]:
                    required_amount = node.contract_demands.get(prod, 0.0)
                 
                    if required_amount > 0.001:
                        # Fluxo gerado = O que mandou pra fora + O que vendeu no mercado interno
                        outbound_flow = pulp.lpSum(outbound_map[node_id][prod])
                        domestic_flow = self.domestic_vars.get((node_id, prod), 0)
                        total_supplied = outbound_flow + domestic_flow
                        
                        slack_name = f"Missed_Production_Quota_{node_id}_{prod.value}".replace("-", "_")
                        slack_contract = pulp.LpVariable(slack_name, lowBound=0)
                        self.contract_slacks[slack_name] = slack_contract
                        
                        self.prob += (total_supplied + slack_contract) >= required_amount, f"Processing_Quota_{node_id}_{prod.value}"

            # --- CASE 4: PORTS ---
            elif node.type == NodeType.PORT:
                for product in ProductType:

                    flow_in = pulp.lpSum(inbound_map[node_id][product])
                    flow_out = pulp.lpSum(outbound_map[node_id][product])

                    stock_final = self.storage_vars.get((node_id, product), 0)
                    initial_inv = node.initial_inventory.get(product.value, 0)
                    
                    target_volume = node.contract_demands.get(product, 0)

                    print('contract ports', product,target_volume)

                    exported_var = pulp.LpVariable(
                        f"Exported_{node_id}_{product.value}", 
                        lowBound=0, 
                        upBound=target_volume 
                    )
                    
                    self.export_vars[(node_id, product)] = exported_var
                    
                    self.prob += (flow_in + initial_inv) == (stock_final + exported_var)
                    self.prob += (flow_out) == 0


            # --- CASE 5: TRAIN STATION ---
            elif node.type == NodeType.TRAIN:
                for product in ProductType:

                    flow_in = pulp.lpSum(inbound_map[node_id][product])
                    flow_out = pulp.lpSum(outbound_map[node_id][product])
                    
                    self.prob += flow_in == flow_out, f"Mass_Balance_Train_{node_id}_{product.value}"


            else:

                for product in ProductType:

                    flow_in = pulp.lpSum(inbound_map[node_id][product])
                    flow_out = pulp.lpSum(outbound_map[node_id][product])
                    
                    stock_final = self.storage_vars.get((node_id, product), 0)
                    initial_inv = node.initial_inventory.get(product.value, 0)
 
                    # Silos/Hubs (Standard behavior)
                    slack_out = self.dummy_sink_vars.get(node_id, 0)
                    
                    self.prob += (flow_in + initial_inv) == (flow_out + stock_final + slack_out)
                    self.prob += stock_final <= node.capacity, f"Max_Stock_{node_id}_{product.value}"

                    if node.type == NodeType.HUB:
                        # Road or Waterway Hubs (Transshipment)
                        self.prob += flow_in <= (node.capacity * 20.0) # Turnover 15x
                        self.prob += stock_final <= (node.capacity * 0.1) # Max 10% remains for the end of the year

                    elif node.type == NodeType.SILO_AGGREGATOR:
                        # The Interior Lungs (Where the real stock stays)
                        self.prob += flow_in <= (node.capacity * 10.0) # Turnover 5x (Soybean harvest + Corn off-season)


            # CASE 6: CONTRACTS - DECISION TREE OUTPUTS 
            for product, required_amount in node.contract_demands.items():
                if required_amount > 0.001:

                    if node.type == NodeType.PROCESSING and product in [ProductType.SOYBEAN_CAKE, ProductType.SOYBEAN_OIL]:
                        continue

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
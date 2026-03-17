# main.py
import pulp, json
from src.domain import SupplyChainNetwork, ProductType, NodeType
from src.optimizer import SupplyChainSolver
from src.loaders import preparation
import polars as pl


# This allows us to use the raw names from the PDF/Excel in our config
ABIOVE_MAPPING = {
    "SOYBEAN": ProductType.SOYBEANS,
    "SOYBEAN_CAKE": ProductType.SOYBEAN_CAKE,
    "SOYBEAN_OIL": ProductType.SOYBEAN_OIL
}

NATIONAL_ESTIMATIVES_ABIOVE = {
    "2022/23": {
        "SOYBEAN": {
            "Initial stock": 3_706_000,     
            "Production": 160_300_000,       
            "Exports": 101_870_000,         
            "Final stock": 5_861_000     
        },
        "SOYBEAN_CAKE": {
            "Initial stock": 2_322_000,      
            "Production": 42_292_000,
            "Exports": 22_474_000,            
            "Domestic sales": 20_511_000,    
            "Final stock": 1_632_00
        },
        "SOYBEAN_OIL": {
            "Initial stock": 520_000,    
            "Production": 10_781_000,
            "Exports": 2_333_000,      
            "Domestic sales": 8_677_000,    
            "Final stock": 312_000
        }
    },
}


# ========================================================
# 1. FINANCIAL AND TECHNICAL PARAMETERS (SCALED)
# ========================================================
YIELD_CAKE = 0.78
YIELD_OIL = 0.19

# Scale down all costs. Instead of millions, use tens and thousands.
# RULE: Penalties MUST be strictly higher than Rewards to prevent infinite loops!

P_CONTRACT = 50_000.0        # High penalty for missing a real contract
P_MAGIC_GEN = 100_000.0      # Massive penalty for breaking physics (must be highest!)
P_DUMMY = 80_000.0           # Penalty for buying/selling out of the system

# Rewards should be small nudges just to prioritize paths, not get-rich-quick schemes
EXPORT_REWARD = -1_000.0     
DOMESTIC_REWARD = -500.0



def model_with_train():
    export_vars = {}

    # Initialize Network    
    network = SupplyChainNetwork()

    # Load input data
    network = preparation(network, paths={
        'production': 'data/production.csv',
        'silos': 'data/silos.csv',
        'ports': 'data/branches.csv',
        'train': 'data/train_flows.csv',
        'fixed_flows': 'data/branches.csv',
        'rail_flows': 'data/train_flows.csv',
        'truck_costs': 'data/cost.csv',
        'industrial_capacity': 'data/industrial_capacity.csv'
    })


    # Adding optmizer
    solver = SupplyChainSolver(name="Soy_Optimization_Final")
    solver.add_flow_variables(network.edges, [p for p in ProductType])

    # Creating Slacks
    industry_ids = [n.id for n in network.nodes.values() if n.type == NodeType.PROCESSING]
    solver.add_supply_slacks(node_ids=industry_ids, penalty_cost=P_DUMMY, product=ProductType.SOYBEANS)

    surplus_nodes = [n.id for n in network.nodes.values() if n.type in [NodeType.PORT, NodeType.SILO_AGGREGATOR, NodeType.PRODUCTION]]
    solver.add_sink_slacks(node_ids=surplus_nodes, penalty_cost=P_DUMMY)

    # Optimzer access to nodes
    solver.prepare_flow_indexes()

    # Transport Costs
    transport_costs = []
    for edge in network.edges:
        for prod in ProductType:
            var = solver.flow_vars.get((edge.source_id, edge.target_id, edge.mode, prod))
            if var: 
                transport_costs.append(var * edge.unit_cost)

    solver.add_to_objective(pulp.lpSum(transport_costs))

    
    # Constraints
    for node_id, node in network.nodes.items():
        if node.type == NodeType.PRODUCTION:
            beans_out = pulp.lpSum(solver.get_outbound_flows(node_id, ProductType.SOYBEANS))
            cake_out = pulp.lpSum(solver.get_outbound_flows(node_id, ProductType.SOYBEAN_CAKE))
            oil_out = pulp.lpSum(solver.get_outbound_flows(node_id, ProductType.SOYBEAN_OIL))
            solver.add_custom_constraint(beans_out == node.production, f"Supply_{node_id}")
            solver.add_custom_constraint((cake_out + oil_out) == 0, f"No_Magic_Gen_{node_id}")


    for node_id, node in network.nodes.items():    
        if node.type in [NodeType.HUB, NodeType.SILO_AGGREGATOR, NodeType.SILO_LOCAL]:       
            for prod in ProductType:
                flow_in = pulp.lpSum(solver.get_inbound_flows(node_id, prod))
                flow_out = pulp.lpSum(solver.get_outbound_flows(node_id, prod))
                slack_out = solver.get_slack_var(node_id, 'sink', prod) # The default '0' helps if it doesn't exist

                solver.add_custom_constraint((flow_in) == (flow_out + slack_out), f"Mass_Bal_{node_id}_{prod.value}")

                if node.type == NodeType.HUB:
                    solver.add_custom_constraint(flow_in <= (node.capacity * 20.0), f"Turnover_Hub_{node_id}_{prod.value}")
                elif node.type == NodeType.SILO_AGGREGATOR:
                    solver.add_custom_constraint(flow_in <= (node.capacity * 10.0), f"Turnover_Agg_{node_id}_{prod.value}")
                elif node.type == NodeType.SILO_LOCAL:
                    solver.add_custom_constraint(flow_in <= (node.capacity * 1.2), f"Turnover_Loc_{node_id}_{prod.value}")


    for node_id, node in network.nodes.items():
        if node.type == NodeType.TRAIN:
            for prod in ProductType:
                flow_in = pulp.lpSum(solver.get_inbound_flows(node_id, prod))
                flow_out = pulp.lpSum(solver.get_outbound_flows(node_id, prod))
                solver.add_custom_constraint(flow_in == flow_out, f"Train_Bal_{node_id}_{prod.value}")


    for node_id, node in network.nodes.items():

        if node.type == NodeType.PROCESSING:

            beans_in = pulp.lpSum(solver.get_inbound_flows(node_id, ProductType.SOYBEANS))
            beans_out = pulp.lpSum(solver.get_outbound_flows(node_id, ProductType.SOYBEANS))

            slack_in_beans = solver.get_slack_var(node_id, 'supply', ProductType.SOYBEANS)
            
            # The Crushing Carousel
            crushed_beans = beans_in + slack_in_beans  
            solver.add_custom_constraint(crushed_beans >= 0, f"Positive_Crush_{node_id}")
            solver.add_custom_constraint(beans_out == 0, f"No_Beans_Out_{node_id}")

            for prod in [ProductType.SOYBEAN_CAKE, ProductType.SOYBEAN_OIL]:
                prod_out = pulp.lpSum(solver.get_outbound_flows(node_id, prod))
                yield_rate = YIELD_CAKE if prod == ProductType.SOYBEAN_CAKE else YIELD_OIL

                # Yield (Supply = Demand)
                solver.add_custom_constraint(
                    #(crushed_beans * yield_rate) + magic  >= (prod_out),
                    (crushed_beans * yield_rate) >= (prod_out),
                    f"Yield_{prod.value}_{node_id}"
                )

                # Quota Rule (Processing Contract)
                required_amount = node.contract_demands.get(prod, 0.0)
                if required_amount > 0.001:
                    slack_quota = pulp.LpVariable(f"Missed_Quota_{node_id}_{prod.value}", lowBound=0)
                    solver.add_to_objective(slack_quota * P_CONTRACT)
                    solver.add_custom_constraint(
                        (prod_out + slack_quota) >= required_amount,
                        f"Processing_Quota_{node_id}_{prod.value}"
                    )


    for node_id, node in network.nodes.items():        
        if node.type == NodeType.PORT:
            for prod in ProductType:
                flow_in = pulp.lpSum(solver.get_inbound_flows(node_id, prod))
                flow_out = pulp.lpSum(solver.get_outbound_flows(node_id, prod))
                target_volume = node.contract_demands.get(prod, 0)

                exported = pulp.LpVariable(f"Exported_{node_id}_{prod.value}", lowBound=0, upBound=target_volume)
                export_vars[(node_id, prod)] = exported
                solver.add_to_objective(exported * EXPORT_REWARD)
                
                solver.add_custom_constraint((flow_in) == (exported), f"Port_Bal_{node_id}_{prod.value}")
                solver.add_custom_constraint(flow_out == 0, f"Port_No_Out_{node_id}_{prod.value}")


    count = 0
    for i, constr in enumerate(network.constraints):
        # Fetches relevant variables by filtering flow_vars
        relevant_vars = [
            var for (src, dst, mode, prod), var in solver.flow_vars.items()
            if src == constr.source_id and dst == constr.target_id and prod == constr.product
        ]
        
        var_name = f"{constr.source_id}_{constr.target_id}".replace("-", "_")
        slack_var = pulp.LpVariable(f"Missed_FixedFlow_{i}_{var_name}", lowBound=0)
        solver.add_to_objective(slack_var * P_CONTRACT)

        expr = pulp.lpSum(relevant_vars) + slack_var
        constr_name = f"FixedFlow_{i}_{var_name}_{constr.product.value}"

        try:
            if constr.type == 'min':
                solver.add_custom_constraint(expr >= constr.volume, constr_name)
            elif constr.type == 'equal':
                solver.add_custom_constraint(expr == constr.volume, constr_name)
            elif constr.type == 'max':
                solver.add_custom_constraint(expr <= constr.volume, constr_name)
            count += 1
        except Exception as e:
            print(e)
            continue

    print(f"Applied {count} Fixed Flow rules.")

    status = solver.solve(msg=1, threads=4)
    print(f"Optimization finished with status: {pulp.LpStatus[status]}")

    # Saving results
    # =========================================================================================
    solution = {
        "status": pulp.LpStatus[status],
        "objective": pulp.value(solver.prob.objective),
        "variables": {v.name: v.varValue for v in solver.prob.variables()}
    }

    with open("data/2023/lp_solution.json", "w") as f: json.dump(solution, f, indent=4)

    TOLERANCE = 0.01

    # 1. Create a helper function to quickly extract data as Tuples
    def extract_active_vars(var_dict, var_type):
        return [
            # Return a Tuple (faster to load into Polars than Dicts)
            (var_type, src, dst, prod.value, var.varValue, var.name)
            for (src, dst, mode, prod), var in var_dict.items()
            # Keep your memory-saving pre-filter here
            if var.varValue and var.varValue > TOLERANCE
        ]

    # 2. Build the Polars DataFrame directly from the combined lists
    df_results = pl.DataFrame(
        extract_active_vars(solver.flow_vars, "FLOW") + extract_active_vars(solver.slack_vars, "SLACK"),
        schema=["type", "node_id", "destination", "product", "volume", "var_solver"]
    )

    df_results.write_parquet('data/2023/lp_solution.parquet')

    # reporter.report_network_retention()


if __name__ == "__main__":
    model_with_train()

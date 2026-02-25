# main.py
import os
from src.domain import SupplyChainNetwork, ProductType
from src.optimizer import SoyChainSolver
from src.reporter import SoyChainReporter
from src.loaders import (
    load_production_nodes, 
    load_silo_nodes, 
    load_port_nodes, 
    load_processing_nodes,
    load_truck_cost_matrix,
    load_fixed_flows_as_constraint,
    load_train_nodes,
    load_distribution_initial_stocks,
    load_train_constraint,
)




# This allows us to use the raw names from the PDF/Excel in our config
ABIOVE_MAPPING = {
    "SOYBEAN": ProductType.SOYBEAN,
    "SOYBEAN_CAKE": ProductType.SOYBEAN_CAKE,
    "SOYBEAN_OIL": ProductType.SOYBEAN_OIL
}

# ---------------------------------------------------------------------------
# SCENARIO CONFIGURATION: "Micro-Scale Test"
# Total System Volume: 2,200 Tons
# ---------------------------------------------------------------------------
# NOTE: The Optimizer usually scales input values by 1000.0.
# So: 2.2 input -> 2,200 tons target.
# ---------------------------------------------------------------------------

ABIOVE_DB = {
    "2022/23": {
        "SOYBEAN": {
            "Initial stock": 3_706_000,      # 500 tons
            "Production": 160_300_000,         # 2,200 tons (Matches FARMs 1000+1200)
            "Exports": 101_870_000,            # 1,200 tons (Target)
            # "Processing": 54_165_000,         # 800 tons (Target)
            "Final stock": 5_861_000         # 0 tons
        },
        "SOYBEAN_CAKE": {
            "Initial stock": 2_322_000,      # 100 tons
            "Production": 42_292_000,
            "Exports": 22_474_000,            # 600 tons (Fits within 624 supply)
            "Domestic sales": 20_511_000,    
            "Final stock": 1_632_00
        },
        "SOYBEAN_OIL": {
            "Initial stock": 520_000,     # 50 tons
            "Production": 10_781_000,
            "Exports": 2_333_000,           # 150 tons (Fits within 152 supply)
            "Domestic sales": 8_677_000,    
            "Final stock": 312_000
        }
    },
}

mapping_abiove = {
    "SOYBEAN": ProductType.SOYBEAN,
    "SOYBEAN_CAKE": ProductType.SOYBEAN_CAKE,
    "SOYBEAN_OIL": ProductType.SOYBEAN_OIL
}

def model_with_train():

    # 0. Load national balances from ABIOVE for evaluation purposes
    # =========================================================================================
    selected_year = "2022/23"
    stats_year = ABIOVE_DB[selected_year]

    # Initialize Network
    network = SupplyChainNetwork()
    print("--- 1. Ingestion Phase ---")

    # 1. Load Nodes (Data 1, 2, 4)
    # =========================================================================================
    load_production_nodes("data/production.csv", network)
    load_silo_nodes("data/silos.csv", network) # LOCAL, AGGREGATOR and HUBS
    load_train_nodes("data/train_flows.csv", network)
    load_port_nodes("data/branches.csv", network)
    load_processing_nodes("data/industrial_capacity.csv", network)


    # 2. Load Edges (Transport Matrix) and Fixed Flows (SEI-PCS)
    # =========================================================================================
    
    # Train station transported volumes
    load_train_constraint('data/train_flows.csv', network)

    # Decision Tree Results (Fixed Contracts - Hard Constraints)
    # load_fixed_flows_edges("data/branches.csv", network)
    load_fixed_flows_as_constraint("data/branches.csv", network)

    # This automatically links the nodes we just loaded
    load_truck_cost_matrix("data/cost.csv", network)

    # 3. For Stocks we use a national value from ABIOVE and distribute amoung facilities
    # =========================================================================================

    # Load initial stocks from ABIOVE
    # For stocks uses a national estimate and distribute based on equivalent factor
    # load_distribution_initial_stocks(network, stats_year)

    # 5. Optimization Process - LP
    # =========================================================================================
    print("\n--- 2. Optimization Phase ---")
    solver = SoyChainSolver(network)
    solver.build_model()
    
    solver.solve()


    # 6. Report results
    # =========================================================================================

    reporter = SoyChainReporter(solver, output_dir="data/2023")

    reporter.generate_validation_summary_report(
        national_stats=stats_year, 
        mapping=mapping_abiove
    )

    reporter.generate_diagnostic_report()

    reporter.export_final_destinations()

    # reporter.report_network_retention()


if __name__ == "__main__":
    model_with_train()

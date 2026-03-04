# main.py
import os
from src.domain import SupplyChainNetwork, ProductType, NodeType
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


def model_with_train():

    # 0. Load national balances from ABIOVE for evaluation purposes
    # =========================================================================================
    selected_year = "2022/23"
    stats_year = NATIONAL_ESTIMATIVES_ABIOVE[selected_year]

    # Initialize Network
    network = SupplyChainNetwork()
    print("--- 1. Ingestion Phase ---")

    # 1. Load Nodes (Data 1, 2, 4)
    # =========================================================================================
    load_production_nodes("data/production.csv", network)
    load_silo_nodes("data/silos.csv", network) # LOCAL, AGGREGATOR and HUBS
    load_processing_nodes("data/industrial_capacity.csv", network)
    load_train_nodes("data/train_flows.csv", network)
    load_port_nodes("data/branches.csv", network)
    
    # 2. Load Edges (Transport Matrix) and Fixed Flows (SEI-PCS)
    # =========================================================================================
    
    # Train station transported volumes
    load_train_constraint('data/train_flows.csv', network)

    # Decision Tree Results (Fixed Contracts - Hard Constraints)
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
        mapping=ABIOVE_MAPPING
    )

    reporter.generate_diagnostic_report()

    reporter.export_final_destinations()

    # reporter.report_network_retention()


if __name__ == "__main__":
    model_with_train()

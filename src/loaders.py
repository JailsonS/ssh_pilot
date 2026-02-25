# src/loaders.py
import pandas as pd
from src.domain import Node, NodeType, SupplyChainNetwork, Edge, ProductType
from collections import defaultdict


DISTANCE_THRESHOULD = 120_000


def load_truck_cost_matrix(file_path: str, network: SupplyChainNetwork):
    print(f"Loading Cost Matrix with Smart-Matching (BR-*) from {file_path}...")
    
    # Lê tudo como string para garantir match exato
    df = pd.read_csv(file_path, delimiter=';')
    df = df[df['cost'] < 40]

    print(f'Número de rotas antes da filtragem {df.shape[0]}')


    # df = df[df['cost'] < 60].copy()
    df['cost'] = df['cost'].mul(3600)

    print(f'Número de rotas após a filtragem {df.shape[0]}')

    # ---------------------------------------------------------
    # PASSO 1: Mapear Nós usando a chave 'BR-...'
    #   - For each municipality get all nodes within it
    # ---------------------------------------------------------
    nodes_by_location = defaultdict(list)
    
    for node in network.nodes.values():

        """
        Lógica Inteligente: Encontra onde começa 'BR-' e pega até o fim.
        Ex: 'SILO_LOCAL-BR-1100023' -> 'BR-1100023'
        Ex: 'PRODUCTION-MT-BR-5107909' -> 'BR-5107909'
        Ex: 'BR-1234567' -> 'BR-1234567'
        """
        keyword = "BR-"
        start_index = node.id.find(keyword)
        
        geo_key = node.id[start_index:] if start_index != -1 else node.id[start_index:]

        nodes_by_location[geo_key].append(node)

    # ---------------------------------------------------------
    # PASSO 2: Iterar a Matriz e Conectar
    # ---------------------------------------------------------
    count = 0
    ignored_no_nodes = 0
    ignored_logic = 0
    
    # Cache para evitar spam no terminal
    missing_origins = set()
    missing_dests = set()

    for _, row in df.iterrows():
        # Na matriz: municipality_origin = "BR-5300108"
        origin_key = str(row['origin']).strip()
        dest_key = str(row['destination']).strip()
        
        try:
            cost = float(row['cost'])
        except (ValueError, TypeError):
            continue

        # Verifica se temos nós nessas chaves
        if origin_key not in nodes_by_location:
            missing_origins.add(origin_key)
        
        if dest_key not in nodes_by_location:
            missing_dests.add(dest_key)
            
        if origin_key not in nodes_by_location or dest_key not in nodes_by_location:
            ignored_no_nodes += 1
            continue

        # Explosão Combinatória: Todos da Origem -> Todos do Destino
        sources = nodes_by_location[origin_key]
        targets = nodes_by_location[dest_key]

        for source_node in sources:
            for target_node in targets:

                if source_node.id == target_node.id:
                    ignored_logic += 1
                    continue
                
                # Validação de Negócio (Fazenda -> Silo, etc)
                if not is_valid_route(source_node.type, target_node.type):
                    ignored_logic += 1
                    continue

                # PRUNING 3: Regra de Eficiência de Distância
                if cost > DISTANCE_THRESHOULD and target_node.type not in [NodeType.PORT, NodeType.TRAIN, NodeType.PROCESSING]:
                    ignored_logic += 1
                    continue
            
                mode = 'rail' if source_node.type == NodeType.TRAIN else 'truck'
                
                edge = Edge(
                    source_id=source_node.id,
                    target_id=target_node.id,
                    mode=mode,
                    unit_cost=cost,
                    max_capacity=1e15 # Capacidade ilimitada 
                )
                network.add_edge(edge)
                count += 1

    print(f"-" * 60)
    print(f"✅ LOAD COMPLETE.")
    print(f"   Edges Created: {count}")
    print(f"   Ignored (Logic/Invalid Route): {ignored_logic}")
    print(f"   Ignored (Locations not found in Nodes): {ignored_no_nodes}")
    
    if missing_origins:
        print(f"   ⚠️  Exemplos de Origens na Matriz sem Nós correspondentes: {list(missing_origins)[:3]}")
    if missing_dests:
        print(f"   ⚠️  Exemplos de Destinos na Matriz sem Nós correspondentes: {list(missing_dests)[:3]}")
    print(f"-" * 60)


def load_production_nodes(file_path: str, network: SupplyChainNetwork):
    """
    Reads Data 1 (Soy Production) and creates PRODUCTION nodes.
    """
    print(f"Loading production data from {file_path}...")
    df = pd.read_csv(file_path, delimiter=';')

    for _, row in df.iterrows():
        node_id = str(row['node_id'])
        # Create Node
        node = Node(
            id=node_id,
            type=NodeType.PRODUCTION,
            production=float(row['volume']),
            inventory_cost=1_000_000 # everything should leave the farm
            # lat=row['lat'], lon=row['lon'] 
        )
        network.add_node(node)


def load_silo_nodes(file_path: str, network: SupplyChainNetwork):
    """
    """
    print(f"Loading silo data from {file_path}...")
    df = pd.read_csv(file_path, delimiter=';')

    for _, row in df.iterrows():
        
        # 1. Dados do Silo
        capacity = float(row['capacity_tons'])
        # cost = float(row.get('cost_per_ton', 5.0))
        silo_type = row.get('facility_type', 'SILO_AGGREGATOR')
        
        # O Código IBGE puro (ex: "5107909")
        trase_id = str(row['node_id'])

        # O ID do Nó Silo (ex: "SILO_AGGREGATOR-BR-5107909")
        silo_node_id = f'{silo_type}-{trase_id}'

        # Cria o Nó Silo
        node = Node(
            id=silo_node_id,
            type=NodeType(silo_type),
            capacity=capacity
        )
        network.add_node(node)


def load_port_nodes(file_path: str, network: SupplyChainNetwork):
    """
    Reads Data 4 (Ports) and creates PORT nodes.
    """
    print(f"Loading port data from {file_path}...")
    df = pd.read_csv(file_path, delimiter=';')

    df_ports = df.groupby(by=['target_id', 'product_type'], as_index=False)['volume'].sum()

    df_ports_pivot = df_ports.pivot(
        index='target_id', 
        columns='product_type', 
        values='volume'
    ).fillna(0).reset_index()
    
    for _, row in df_ports_pivot.iterrows():
        node_id = str(row['target_id'])

        vol_cake = float(pd.to_numeric(row.get(ProductType.SOYBEAN_CAKE.value, 0), errors="coerce") or 0)
        vol_oil = float(pd.to_numeric(row.get(ProductType.SOYBEAN_OIL.value, 0), errors="coerce") or 0)
        vol_bean = float(pd.to_numeric(row.get(ProductType.SOYBEAN.value, 0), errors="coerce") or 0)

        volume_total = vol_bean + vol_cake + vol_oil

        node = Node(
            id=node_id,
            type=NodeType.PORT,
            capacity=volume_total,
            contract_demands={
                ProductType.SOYBEAN: vol_bean,
                ProductType.SOYBEAN_CAKE: vol_cake,
                ProductType.SOYBEAN_OIL: vol_oil
            }
        )
        network.add_node(node)


def load_train_nodes(file_path: str, network: SupplyChainNetwork):
    """
    """
    print(f"Loading port data from {file_path}...")
    df = pd.read_csv(file_path, delimiter=';')

    df_stations = pd.DataFrame(
        pd.concat([df['origin_id'], df['dest_id']]).unique(),
        columns=['node_id']
    )

    for _, row in df_stations.iterrows():
        node_id = str(row['node_id'])
        
        node = Node(
            id=node_id,
            type=NodeType.TRAIN,
            capacity=1_000_000.,
        )
        network.add_node(node)


def load_processing_nodes(file_path: str,  network: SupplyChainNetwork):
    """
    Reads Data 5 (ABIOVE Processing Capacity) and creates PROCESSING nodes.
    """
    print(f"Loading processing data from {file_path}...")
    df = pd.read_csv(file_path, delimiter=';')

    # use CONAB national soy balances to calculate quantity of soy crushed
    # calulate soybean volume equivalent to crushing products using equivalence factor
    NATIONAL_VOL_CAKE = 42_292_000
    NATIONAL_VOL_OIL = 10_781_000

    total_capacity = df["capacity_tons"].sum() * 1.031

    df["vol_processed_beans_for_cake"] = (
        df["capacity_tons"] * NATIONAL_VOL_CAKE / total_capacity * 1.031
    )

    df["vol_processed_beans_for_oil"] = (
        df["capacity_tons"] * NATIONAL_VOL_OIL / total_capacity * 1.031
    )
    
    for _, row in df.iterrows():
        node_id = str(row['node_id'])
        capacity = float(row['capacity_tons'])
        capacity_cake = float(row.get('vol_processed_beans_for_cake', 0))
        capacity_oil = float(row.get('vol_processed_beans_for_oil', 0))

        node = Node(
            id=node_id,
            type=NodeType.PROCESSING,
            capacity=capacity,
            inventory_cost=3.0,
            contract_demands={
                ProductType.SOYBEAN_CAKE: capacity_cake,
                ProductType.SOYBEAN_OIL: capacity_oil
            }
        )
        network.add_node(node)


def load_train_constraint(file_path: str, network: SupplyChainNetwork):
    """
    Reads Data 3 (Railway Flows) and adds them as RAIL edges.
    
    Assumption: The CSV contains 'origin_id', 'dest_id', 'volume', 'tariff_cost'.
    """
    print(f"Loading rail data from {file_path}...")
    df = pd.read_csv(file_path, delimiter=';')
    
    count = 0
    for _, row in df.iterrows():
        origin = str(row['origin_id'])
        dest = str(row['dest_id'])
        volume = float(row['volume'])
        
        # Parse Product Type
        prod_str = str(row['product_type']).strip().upper()
        product_enum = ProductType(prod_str) 
        
        # Validation: Ensure stations exist as Nodes 
        if origin in network.nodes and dest in network.nodes:

            network.add_constraint(
                source_id=origin,
                target_id=dest,
                product=product_enum,
                volume=volume,
                type='equal' 
            )

            # edge = Edge(
            #     source_id=origin,
            #     target_id=dest,
            #     mode='train',
            #     unit_cost=5,
            #     fixed_flow=volume,
            #     fixed_product=product_enum
            # )

            # network.add_edge(edge)

            count += 1
        else:
            # Optional: Warning for missing nodes
            # print(f"Warning: Rail stations {origin} or {dest} not found in network.")
            pass
            
    print(f"Successfully added {count} rail segments.")


def load_fixed_flows_as_constraint(file_path: str,  network: SupplyChainNetwork):
    """
    Reads fixed flows including PRODUCT_TYPE.
    CSV Columns: source_id, target_id, volume, mode, product_type
    """
    
    df = pd.read_csv(file_path, delimiter=';')

    # 1. filter to branches
    df = df[df['branch'].str.startswith(('1.', '2.', '3.'))].copy()

    # 2. adjust source id. 
    mask_1 = df['branch'].str.startswith('1.')
    df.loc[mask_1, 'source_id'] = df.loc[mask_1, 'source_id'].str.replace('PORT', 'PRODUCTION')

    mask_2 = df['branch'].str.startswith('2.')
    df.loc[mask_2, 'source_id'] = df.loc[mask_2, 'source_id'].str.replace('PORT', 'HUB')
    
    mask_3 = df['branch'].str.startswith('3.')
    df.loc[mask_3, 'source_id'] = df.loc[mask_3, 'source_id'].str.replace('PORT', 'PROCESSING')

    df = df[['product_type', 'source_id', 'target_id', 'port_of_export_name', 'volume']].groupby(
        ['product_type', 'source_id', 'target_id', 'port_of_export_name'],
        as_index=False
    )['volume'].sum()

    for _, row in df.iterrows():        
        # Parse Product Type
        prod_str = str(row['product_type']).strip()
        try:
            # Map "Soybean" -> ProductType.SOYBEAN
            product_enum = ProductType(prod_str) 
        except ValueError:
            print(f"Warning: Unknown product {prod_str}, skipping.")
            continue
        
        # Simplificando o acesso aos IDs
        source_id = str(row['source_id'])
        target_id = str(row['target_id'])

        # Descobre os NodeTypes baseados no prefixo (Ex: "PRODUCTION-123" -> "PRODUCTION")
        # Adicionado um bloco try/except caso venha algum ID fora do padrão mapeado no Enum
        try:
            node_type_source = NodeType(source_id.split('-')[0])
            node_type_target = NodeType(target_id.split('-')[0])
        except ValueError as e:
            print(f"Warning: Falha ao identificar o tipo de nó para {source_id} ou {target_id}. Detalhes: {e}")
            continue

        # --- CRIAÇÃO CONDICIONAL DOS NÓS ---
        
        # Verifica se o source já existe na rede; se não, cria e adiciona
        if source_id not in network.nodes:
            network.add_node(Node(id=source_id, type=node_type_source))
            
        # Verifica se o target já existe na rede; se não, cria e adiciona
        if target_id not in network.nodes:
            network.add_node(Node(id=target_id, type=node_type_target))

        # Registra a obrigação de fluxo
        network.add_constraint(
            source_id=source_id,
            target_id=target_id,
            product=product_enum,
            volume=float(row['volume']),
            type='min'
        )


def load_distribution_initial_stocks(network: SupplyChainNetwork, abiove_reference:dict):
    """
    Distributes national initial stock (carry-over) into Silos 
    proportional to their static capacity.
    UPDATED: Assigns a Dictionary {'Soybean': value} instead of a float.
    """
    
    # 1. Calculate Total System Capacity
    total_capacity = sum(
        n.capacity for n in network.nodes.values() 
        if n.type in [NodeType.SILO_AGGREGATOR, NodeType.SILO_LOCAL, NodeType.HUB]
    )
    
    if total_capacity == 0:
        print("WARNING: No Silo capacity found to store initial stock.")
        return
    
    total_initial_stock_beans_tons = abiove_reference[ProductType.SOYBEAN.value]['Initial stock']
    total_initial_stock_cake_tons = abiove_reference[ProductType.SOYBEAN_CAKE.value]['Initial stock'] 
    total_initial_stock_oil_tons = abiove_reference[ProductType.SOYBEAN_OIL.value]['Initial stock']

    print(f'Initial Bean Stocks {total_initial_stock_beans_tons:.0f}')
    print(f'Initial Cake Stocks {total_initial_stock_cake_tons:.0f}')
    print(f'Initial Oil Stocks {total_initial_stock_oil_tons:.0f}')

    # 2. Distribute
    count = 0
    for node in network.nodes.values():
        if node.type in [NodeType.SILO_AGGREGATOR, NodeType.SILO_LOCAL, NodeType.HUB]:

            # Ratio of this silo's capacity to total
            ratio = node.capacity / total_capacity
            
            # Calculate tonnage
            stock_val = total_initial_stock_beans_tons * ratio
            
            # --- FIXED: Assign a Dictionary, defaulting to SOYBEAN ---
            # Old Code: node.initial_inventory = stock_val
            node.initial_inventory = {
                ProductType.SOYBEAN.value: stock_val
            }
            
            count += 1

        if node.type in [NodeType.PROCESSING]:
            # Ratio of this silo's capacity to total
            ratio = node.capacity / total_capacity
            
            # Calculate tonnage
            stock_cake_val = total_initial_stock_cake_tons * ratio
            stock_oil_val = total_initial_stock_oil_tons * ratio
            
            # --- FIXED: Assign a Dictionary, defaulting to SOYBEAN ---
            # Old Code: node.initial_inventory = stock_val
            node.initial_inventory = {
                ProductType.SOYBEAN_CAKE.value: stock_cake_val,
                ProductType.SOYBEAN_OIL.value: stock_oil_val
            }       

    print(f"Allocated initial stock to {count} silos.")


def is_valid_route(src_type: NodeType, dst_type: NodeType) -> bool:
    """
    Define as regras de negócio: O que pode conectar com o que?
    """
    
    # REGRA 1: SILOS (Concentradores de Grãos)
    # Podem mandar para: Outros Silos, Portos, Indústrias
    if src_type in [NodeType.SILO_AGGREGATOR, NodeType.SILO_LOCAL, NodeType.HUB]:
        if dst_type in [
            NodeType.SILO_AGGREGATOR, 
            NodeType.PORT, 
            NodeType.PROCESSING,
            NodeType.HUB,
            NodeType.TRAIN
        ]:
            return True

    # REGRA 2: INDÚSTRIA (Processadores)
    # Podem mandar para: Portos (Exportar Óleo/Farelo) ou Silos (raro, mas possível para estocar)
    # ou Mercados Domésticos (se você tiver esse nó)
    if src_type == NodeType.PROCESSING:
        if dst_type in [
            NodeType.PORT,
            NodeType.HUB,
            NodeType.TRAIN
        ]:
            return True

    # REGRA 3: FAZENDAS (Production)
    # Geralmente fazendas mandam para o Silo LOCAL (intra-municipio).
    # Mas se quiser permitir envio direto para Silo ou Porto em OUTRO município:
    if src_type == NodeType.PRODUCTION:
        # Permite que a fazenda entregue em qualquer tipo de Silo ou Hub
        if dst_type in [
            NodeType.PROCESSING, 
            NodeType.HUB,  
            NodeType.PORT,
            NodeType.TRAIN
        ]:
            return True

    if src_type == NodeType.TRAIN:
        # O trem pode descarregar em:
        # 1. Outra estação de trem (conexão de trilhos)
        # 2. Porto (Exportação)
        # 3. Indústria (Para esmagamento)
        # 4. Silo/Hub (Transbordo)
        if dst_type in [NodeType.TRAIN, NodeType.PORT, NodeType.PROCESSING, NodeType.HUB]:
            return True
    return False

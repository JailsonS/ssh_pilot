# src/reporter.py
import pandas as pd
import os
from src.domain import NodeType, ProductType
from collections import defaultdict


class SoyChainReporter:
    def __init__(self, solver, output_dir="output"):
        self.solver = solver
        self.network = solver.network
        self.output_dir = output_dir
        self.flow_graph = None 
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)


    def generate_validation_summary_report(self, national_stats: dict, mapping: dict):
        """Flexible validator. CORRIGIDO PARA LER EXPORT_VARS."""
        print("\n" + "="*80)
        print(f"{'ABIOVE VALIDATION SUMMARY':^80}")
        print("="*80)

        actuals = {
            "Exports": defaultdict(float),
            "Processing": defaultdict(float),
            "Final_Stock": defaultdict(float)
        }

        # --- CORREÇÃO 1: Ler EXPORTAÇÕES da variável correta (Saída do Sistema) ---
        # Não olhe para o fluxo de caminhão, olhe para a variável de Exportação do Porto.
        # Usa getattr para evitar erro se a variável ainda não existir
        export_vars = getattr(self.solver, 'export_vars', {})
        
        for (node_id, prod), var in export_vars.items():
            if var.varValue and var.varValue > 0.01:
                actuals["Exports"][prod] += var.varValue

        # --- CORREÇÃO 2: PROCESSAMENTO (Esmagamento) ---
        # Para "Processing", a ABIOVE geralmente quer saber quanto de GRÃO entrou na fábrica.
        # Se você quiser medir produção de óleo, olhe o output. 
        # Vou manter sua lógica de fluxo de entrada, mas filtrando para SOYBEAN (Input da fábrica).
        for (src, dst, mode, prod), var in self.solver.flow_vars.items():
            if var.varValue and var.varValue > 0.01:
                # Proteção: verifica se o nó existe na rede (pode ser um nó auxiliar)
                if dst in self.network.nodes:
                    dst_type = self.network.nodes[dst].type
                    
                    if dst_type == NodeType.PROCESSING:
                        # O "Processing" alvo geralmente é a entrada de GRÃOS para esmagamento
                        if prod == ProductType.SOYBEANS:
                            actuals["Processing"][prod] += var.varValue
                        
                        # Se você quiser medir "Produção Industrial" (quanto gerou de óleo):
                        # Você precisaria somar o 'total_input * yield' ou o flow de saída.
                        # Mas para comparar com 'Processing' (Esmagamento), o input de grão é o correto.

        # --- CORREÇÃO 3: ESTOQUES ---
        for (node_id, prod), var in self.solver.storage_vars.items():
            if var.varValue and var.varValue > 0.01:
                actuals["Final_Stock"][prod] += var.varValue

        # Função auxiliar de formatação (mantida igual)
        def fmt_row(metric, target, actual_val):
            target_t = target / 1_000_000
            actual_t = actual_val / 1_000_000
            
            # Evita divisão por zero
            if target_t > 0:
                diff_pct = ((actual_t - target_t) / target_t) * 100 
            elif actual_t > 0:
                diff_pct = 100.0 # Target 0 mas teve actual
            else:
                diff_pct = 0.0 # Ambos 0

            print(f"    {metric:<15} | {target_t:10,.1f} Mt | {actual_t:10,.1f} Mt | {diff_pct:+6.1f}%")

        # Loop de Impressão (Mantido igual)
        for abiove_name, targets in national_stats.items():
            if abiove_name not in mapping: continue
            
            prod_enum = mapping[abiove_name]
            print(f"\n--- PRODUCT: {abiove_name.upper()} ({prod_enum.value}) ---")
            print(f"    {'Metric':<15} | {'Target':>13} | {'Actual':>13} | {'Diff %':>7}")
            print("-" * 60)

            if "Exports" in targets:
                fmt_row("Exports", targets["Exports"], actuals["Exports"][prod_enum])

            if "Processing" in targets:
                # A métrica de "Processing" só faz sentido para o GRÃO (Input).
                # Para óleo/farelo, o dado seria "Production", que não estamos medindo aqui.
                if prod_enum == ProductType.SOYBEANS:
                     fmt_row("Processing", targets["Processing"], actuals["Processing"][prod_enum])
                else:
                     # Se for Óleo/Farelo, Processing = 0 (eles são output, não input de esmagamento)
                     pass

            if "Final stock" in targets:
                fmt_row("Final Stock", targets.get("Final stock", 0), actuals["Final_Stock"][prod_enum])


    def report_network_retention(self):
        print("\n" + "="*80)
        print(f"{'GLOBAL SUPPLY CHAIN RETENTION REPORT':^80}")
        print("="*80)
        
        # Dicionário para acumular por tipo: {NodeType: {Product: Volume}}
        retention_by_type = defaultdict(lambda: defaultdict(float))
        total_by_product = defaultdict(float)

        for (node_id, product), var in self.solver.storage_vars.items():
            volume = var.varValue if var.varValue is not None else 0
            if volume > 0.01:
                node_type = self.network.nodes[node_id].type.value
                retention_by_type[node_type][product.value] += volume
                total_by_product[product.value] += volume

        # 1. VISÃO RESUMIDA POR TIPO DE NÓ
        print(f"{'NODE TYPE':<20} | {'PRODUCT':<15} | {'RETAINED (Tons)':>15} | {'% OF TOTAL'}")
        print("-" * 80)
        
        for n_type, products in sorted(retention_by_type.items()):
            for prod_name, vol in products.items():
                share = (vol / total_by_product[prod_name] * 100) if total_by_product[prod_name] > 0 else 0
                print(f"{n_type:<20} | {prod_name:<15} | {vol:>15.2f} | {share:>9.1f}%")
        
        print("-" * 80)
        for prod_name, total in total_by_product.items():
            print(f"TOTAL RETAINED [{prod_name}]: {total/1e6:>36.2f} Mt")

        # 2. IDENTIFICAÇÃO DE GARGALOS (TOP 10 NÓS)
        print("\n" + "-"*30 + " TOP 10 STORAGE GARGALOS " + "-"*30)
        node_details = []
        for (node_id, product), var in self.solver.storage_vars.items():
            vol = var.varValue or 0
            if vol > 0.1:
                node_details.append((node_id, product.value, vol))
        
        node_details = sorted(node_details, key=lambda x: x[2], reverse=True)
        for i, (n_id, prod, vol) in enumerate(node_details[:10], 1):
            print(f"{i:2}. {n_id:<40} | {prod:<12} | {vol:>12.2f} tons")
        
        print("="*80)


    def export_final_destinations(self, filename="balanco_destinos_finais.csv"):
        print("\nExtraindo Destinos Finais (Sem dupla contagem)...")
        records = []
        TOLERANCE = 0.01

        # 1. EXPORTAÇÃO (O que efetivamente saiu do país pelos Portos)
        if hasattr(self.solver, 'export_vars'):
            for (node_id, prod), var in self.solver.export_vars.items():
                if var.varValue and var.varValue > TOLERANCE:
                    records.append({
                        "Categoria": "EXPORTACAO",
                        "Node_ID_Destino": node_id,
                        "Produto": prod.value,
                        "Volume_Tons": var.varValue,
                        "Volume_Mt": var.varValue / 1_000_000 # Ajuda a ler os milhões
                    })

        # 2. MERCADO DOMÉSTICO (O que foi consumido nas Indústrias)
        if hasattr(self.solver, 'domestic_vars'):
            for (node_id, prod), var in self.solver.domestic_vars.items():
                if var.varValue and var.varValue > TOLERANCE:
                    records.append({
                        "Categoria": "DOMESTICO",
                        "Node_ID_Destino": node_id,
                        "Produto": prod.value,
                        "Volume_Tons": var.varValue,
                        "Volume_Mt": var.varValue / 1_000_000
                    })

        # 3. ESTOQUE FINAL (O que ficou guardado aguardando o próximo ano)
        if hasattr(self.solver, 'storage_vars'):
            for (node_id, prod), var in self.solver.storage_vars.items():
                if var.varValue and var.varValue > TOLERANCE:
                    records.append({
                        "Categoria": "ESTOQUE_RETIDO",
                        "Node_ID_Destino": node_id,
                        "Produto": prod.value,
                        "Volume_Tons": var.varValue,
                        "Volume_Mt": var.varValue / 1_000_000
                    })
                    
        # 4. DESCARTE (Caso o modelo ainda precise jogar algo fora por falta de logística)
        if hasattr(self.solver, 'waste_vars'):
            for (node_id, prod), var in self.solver.waste_vars.items():
                if var.varValue and var.varValue > TOLERANCE:
                    records.append({
                        "Categoria": "LIXO_GARGALO",
                        "Node_ID_Destino": node_id,
                        "Produto": prod.value,
                        "Volume_Tons": var.varValue,
                        "Volume_Mt": var.varValue / 1_000_000
                    })

        # --- Salvar em CSV ---
        df = pd.DataFrame(records)
        
        import os
        filepath = os.path.join(getattr(self, 'output_dir', ''), filename)
        df.to_csv(filepath, index=False)
        
        # --- Imprimir o Resumo de Prova Real ---
        print("\n========================================================")
        print(" PROVA REAL: SOMA TOTAL POR CATEGORIA E PRODUTO (em Mt)")
        print("========================================================")
        if not df.empty:
            summary = df.groupby(['Categoria', 'Produto'])['Volume_Mt'].sum().reset_index()
            # Formata a tabela bonitinha no console
            print(summary.to_string(index=False, float_format="%.2f"))
        else:
            print("Nenhum dado encontrado.")
        print("========================================================")
        print(f"✅ CSV salvo em: {filepath}")
        
        return df


    def generate_diagnostic_report(self, filename="relatorio_auditoria_solver.csv"):
        """
        Gera um relatório completo de todas as variáveis ativas (> 0).
        Separa fluxos reais de variáveis de ajuste (Slacks/Dummies).
        """
        print("\n" + "="*60)
        print(f"{'GERANDO RELATÓRIO DE AUDITORIA E DIAGNÓSTICO':^60}")
        print("="*60)
        
        records = []
        TOLERANCE = 0.01  # Ignora sujeira numérica (ex: 0.0000001)

        # ==============================================================================
        # 0. FLUXOS REAIS (Logística)
        # ==============================================================================
        for (node_id, prod), var in self.solver.waste_vars.items():
            if var.varValue and var.varValue > TOLERANCE:
                records.append({
                    "cateogory": "[ALERT_SLACK]",
                    "type": "WASTE",
                    "node_id": node_id,
                    "destination":'WASTE',
                    "product": prod.value,
                    "volume": var.varValue,
                    "var_solver": var.name
                })

        # ==============================================================================
        # 1. FLUXOS REAIS (Logística)
        # ==============================================================================
        for (src, dst, mode, prod), var in self.solver.flow_vars.items():
            if var.varValue and var.varValue > TOLERANCE:
                records.append({
                    "category": "REAL_LOGISTIC",
                    "type": "FLOW",
                    "node_id": src,
                    "destination": dst,
                    "product": prod.value,
                    "volume": var.varValue,
                    "var_solver": var.name
                })

        # ==============================================================================
        # 2. ESTOQUES (Armazenagem)
        # ==============================================================================
        for (node_id, prod), var in self.solver.storage_vars.items():
            if var.varValue and var.varValue > TOLERANCE:
                records.append({
                    "category": "REAL_ESTOCK",
                    "type": "FINAL_STOCK",
                    "node_id": node_id,
                    "destination": node_id,
                    "product": prod.value,
                    "volume": var.varValue,
                    "var_solver": var.name
                })

        # ==============================================================================
        # 3. EXPORTAÇÕES (Faturamento)
        # ==============================================================================
        # Usa getattr para evitar erro caso a variável não tenha sido criada
        for (node_id, prod), var in getattr(self.solver, 'export_vars', {}).items():
            if var.varValue and var.varValue > TOLERANCE:
                records.append({
                    "category": "REAL_EXPORTS",
                    "type": "EXPORTS",
                    "node_id": node_id,
                    "destination": "INTERNAL",
                    "product": prod.value,
                    "volume": var.varValue,
                    "var_solver": var.name
                })

        # ==============================================================================
        # 4. SLACKS E DUMMIES (AQUI ESTÃO OS PROBLEMAS)
        # ==============================================================================
        
        # 4.1 Dummy Supply (Faltou produto na entrada)
        for node_id, var in self.solver.dummy_supply_vars.items():
            if var.varValue and var.varValue > TOLERANCE:
                records.append({
                    "category": "[ALERT_SLACK]",
                    "type": "LACK_INPUT (MAGIC SUPPLY)",
                    "node_id": node_id,
                    "destination": "DUMMY SUPPLY",
                    "product": "GENERIC",
                    "volume": var.varValue,
                    "var_solver": var.name
                })

        # 4.2 Dummy Sink (Sobrou produto e não tem onde por)
        for node_id, var in self.solver.dummy_sink_vars.items():
            if var.varValue and var.varValue > TOLERANCE:
                records.append({
                    "category": "[ALERT_SLACK]",
                    "type": "OVERFLOW (WASTER)",
                    "node_id": node_id,
                    "destination": "WAST",
                    "product": "GENERIC",
                    "volume": var.varValue,
                    "var_solver": var.name
                })

        # 4.3 Slacks de Processamento (Yield quebrado)
        for node_id, (var_cake, var_oil) in self.solver.processing_slacks.items():
            
            # Checa Slack de Farelo
            if var_cake.varValue and var_cake.varValue > TOLERANCE:
                records.append({
                    "category": "[ALERT_SLACK]",
                    "type": "YIELD_VIOLATION (CAKE)",
                    "node_id": node_id,
                    "destination": "INDUSTRY",
                    "product": "SOYBEAN_CAKE",
                    "volume": var_cake.varValue,
                    "var_solver": var_cake.name
                })
                
            # Checa Slack de Óleo
            if var_oil.varValue and var_oil.varValue > TOLERANCE:
                records.append({
                    "category": "[ALERT_SLACK]",
                    "type": "YIELD_VIOLATION (OIL)",
                    "node_id": node_id,
                    "destination": "INDUSTRY",
                    "product": "SOYBEAN_OIL",
                    "volume": var_oil.varValue,
                    "var_solver": var_oil.name
                })

        # 4.4 Slacks de Contrato (Se existirem)
        contract_slacks = getattr(self.solver, 'contract_slacks', {})
        for name, var in contract_slacks.items():
            if var.varValue and var.varValue > TOLERANCE:
                # Tenta extrair o ID do nó do nome da variável (ex: Missed_Contract_NODEID_SOYBEAN)
                parts = name.split("_")
                prod_guess = parts[-1]
                records.append({
                    "category": "[ALERT_SLACK]",
                    "type": "CONTRACT BROKEN",
                    "node_id": name, # Ou tente parsear o ID
                    "destination": "Multa Paga",
                    "product": prod_guess,
                    "volume": var.varValue,
                    "var_solver": var.name
                })

        # ==============================================================================
        # 5. DOMESTIC DEMAND
        # ==============================================================================
        
        # Domestic
        for (node_id, prod), var in self.solver.domestic_vars.items():
            if var.varValue and var.varValue > TOLERANCE:
                records.append({
                    "category": "DOMESTIC DEMAND",
                    "type": "STOCK/DOMESTIC",
                    "node_id": node_id,
                    "destination": "INTERNAL",
                    "product": prod.value,
                    "volume": var.varValue,
                    "var_solver": var.name
                })

        # ==============================================================================
        # 5. GERAÇÃO DO DATAFRAME E RESUMO
        # ==============================================================================
        df = pd.DataFrame(records)
        
        if df.empty:
            print("⚠️ O modelo rodou mas NENHUMA variável tem valor > 0.01. Verifique se tudo é zero.")
            return df

        # Salva CSV
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False, sep=';')
        
        # --- IMPRESSÃO DO RESUMO NO TERMINAL ---
        print(f"\n✅ Relatório detalhado salvo em: {filename}")
        print("\n--- RESUMO DE SLACKS (ONDE O MODELO FALHOU) ---")
        
        slacks = df[df['category'] == '[ALERT_SLACK]']
        
        if slacks.empty:
            print("🎉 PARABÉNS! NENHUM SLACK UTILIZADO. O modelo é Feasible e Realista.")
        else:
            # Agrupa por subtipo para mostrar onde está o problema
            summary = slacks.groupby(['type', 'product'])['volume'].sum().reset_index()
            summary['estimatided_cost'] = summary.apply(
                lambda x: x['volume'] * (self.solver.P_DUMMY if 'SUPPLY' in x['type'] else self.solver.P_MAGIC_GEN), axis=1
            )
            summary['volume'] = summary['volume'].map('{:,.0f}'.format)
            summary['estimatided_cost'] = summary['estimatided_cost'].map('R$ {:,.2f}'.format)
            
            print(summary.to_string(index=False))
            print("\n⚠️  ATENÇÃO: Valores acima indicam que o modelo precisou 'inventar' ou 'jogar fora' soja.")
            
        return df
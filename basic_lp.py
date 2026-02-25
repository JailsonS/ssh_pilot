import pulp

# 1️⃣ Criar o problema (Maximização)
modelo = pulp.LpProblem("Problema_de_Producao", pulp.LpMaximize)

# 2️⃣ Criar variáveis de decisão
x_A = pulp.LpVariable("Produto_A", lowBound=0)
x_B = pulp.LpVariable("Produto_B", lowBound=0)

# 3️⃣ Definir função objetivo
modelo += 20*x_A + 30*x_B, "Lucro_Total"

# 4️⃣ Adicionar restrições
modelo += 2*x_A + 4*x_B <= 100, "Horas_Maquina"
modelo += 3*x_A + 2*x_B <= 90, "Materia_Prima"

# 5️⃣ Resolver
modelo.solve()

# 6️⃣ Mostrar resultados
print("Status:", pulp.LpStatus[modelo.status])
print("Quantidade A:", x_A.varValue)
print("Quantidade B:", x_B.varValue)
print("Lucro máximo:", pulp.value(modelo.objective))


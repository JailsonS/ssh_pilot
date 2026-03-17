import pulp

# 1️⃣ Create Problem
modelo = pulp.LpProblem("Problema_de_Producao", pulp.LpMaximize)

# 2️⃣ Create variables
x_A = pulp.LpVariable("Produto_A", lowBound=0)
x_B = pulp.LpVariable("Produto_B", lowBound=0)

# 3️⃣ Define objective function
modelo += 20*x_A + 30*x_B, "Lucro_Total"

# 4️⃣ Add constrains
modelo += 2*x_A + 4*x_B <= 100, "Horas_Maquina"
modelo += 3*x_A + 2*x_B <= 90, "Materia_Prima"

# 5️⃣ Solver
modelo.solve()

# 6️⃣ Show results
print("Status:", pulp.LpStatus[modelo.status])
print("Quantidade A:", x_A.varValue)
print("Quantidade B:", x_B.varValue)
print("Lucro máximo:", pulp.value(modelo.objective))


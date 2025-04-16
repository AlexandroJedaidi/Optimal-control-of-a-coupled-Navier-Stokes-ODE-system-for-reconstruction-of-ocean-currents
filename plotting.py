import numpy as np
import matplotlib.pyplot as plt

def generate_dotted_style(k):
    base = k + 1  # Increase the base to vary patterns
    return (0, (base, base // 2))

J_array = []
with open(f"results/dolfin/OCP/experiments/{191}/J_array.npy", "rb") as ud_reader:
    J_array.append(np.load(ud_reader))
with open(f"results/dolfin/OCP/experiments/{190}/J_array.npy", "rb") as ud_reader:
    J_array.append(np.load(ud_reader))
with open(f"results/dolfin/OCP/experiments/{182}/J_array.npy", "rb") as ud_reader:
    J_array.append(np.load(ud_reader))


plt.xlabel(r"Iteration")
plt.ylabel(r"Cost")
# plt.yscale("log")
plt.title(r"Reduced cost $j(q)$")
for i, J in enumerate(J_array):
    if i == 0 :
        labell = 10
    elif i == 1 :
        labell = 100
    elif i == 2 :
        labell = 400
    linestyle = generate_dotted_style(2*i)
    plt.plot(J, color="b", linestyle=linestyle, label=rf"$j_{{{labell}}}$")
plt.legend(loc="best")
plt.savefig(f"results/dolfin/OCP/common_plots/4/J.png")
plt.clf()
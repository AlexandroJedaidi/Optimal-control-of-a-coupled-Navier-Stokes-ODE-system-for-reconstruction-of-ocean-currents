import numpy as np
import matplotlib.pyplot as plt

def generate_dotted_style(k):
    base = k + 1  # Increase the base to vary patterns
    return (0, (base, base // 2))

J_array = []
with open(f"results/dolfin/OCP/experiments/{150}/J_array.npy", "rb") as ud_reader:
    J_array.append(np.load(ud_reader))
with open(f"results/dolfin/OCP/experiments/{151}/J_array.npy", "rb") as ud_reader:
    J_array.append(np.load(ud_reader))
with open(f"results/dolfin/OCP/experiments/{152}/J_array.npy", "rb") as ud_reader:
    J_array.append(np.load(ud_reader))

plt.xlabel(r"Iteration")
plt.ylabel(r"Cost")
# plt.yscale("log")
plt.title(r"Reduced cost $j(q)$")
for i, J in enumerate(J_array):
    linestyle = generate_dotted_style(2*i)
    plt.plot(J, color="b", linestyle=linestyle, label=rf"$j_{i*2+2}$")
plt.legend(loc="best")
plt.savefig(f"results/dolfin/OCP/common_plots/1/J.png")
plt.clf()
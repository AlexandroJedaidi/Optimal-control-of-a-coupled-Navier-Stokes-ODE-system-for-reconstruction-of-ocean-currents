import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
def generate_dotted_style(k):
    base = k + 1  # Increase the base to vary patterns
    return (0, (base, base // 2))

J_array = []
s = ["240_LS", "241_f_LR_5", "242_f_LR_2", "243_f_LR_1.5"]
# s = ["300", "301", "303"]
with open(f"results/dolfin/OCP/experiments/{s[0]}/J_array.npy", "rb") as ud_reader:
    J_array.append(np.load(ud_reader))
with open(f"results/dolfin/OCP/experiments/{s[1]}/J_array.npy", "rb") as ud_reader:
    J_array.append(np.load(ud_reader))
with open(f"results/dolfin/OCP/experiments/{s[2]}/J_array.npy", "rb") as ud_reader:
    J_array.append(np.load(ud_reader))
with open(f"results/dolfin/OCP/experiments/{s[3]}/J_array.npy", "rb") as ud_reader:
    J_array.append(np.load(ud_reader))


plt.xlabel(r"Iteration")
plt.ylabel(r"Cost")
# plt.yscale("log")
plt.title(r"Reduced cost $j(q)$")
for i, J in enumerate(J_array):
    if i == 0 :
        labell = "LS"
        a = 0
    elif i == 1 :
        labell = 5
        a = 2
    elif i == 2 :
        labell = 2
        a = 5
    elif i == 3 :
        labell = 1.5
        a = 13
    linestyle = generate_dotted_style(a)
    plt.plot(J, color="b", linestyle=linestyle, label=rf"$j_{{{labell}}}$")
plt.legend(loc="best")
plt.savefig(f"results/dolfin/OCP/common_plots/3/J.png")
plt.clf()
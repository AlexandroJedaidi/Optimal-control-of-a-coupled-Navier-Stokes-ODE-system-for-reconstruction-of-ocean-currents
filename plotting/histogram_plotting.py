import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300

plt.rcParams["font.family"] = "TeX Gyre Heros"
plt.rcParams["mathtext.fontset"] = "cm"

buoy_counts = ["10", "100", "400", "10000"]
iteration_times = [0.10, 11.98, 77.82, 1500]

plt.figure(figsize=(8, 5))
plt.bar(buoy_counts, iteration_times, color='steelblue', width=0.6)

# Labels and title
plt.xlabel('Number of Buoys')
plt.yscale("log")
plt.ylabel('Average time per Iteration (seconds)')
plt.title('Average iteration Time vs Number of Buoys')
plt.xticks(buoy_counts)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(f"results/dolfin/OCP/common_plots/timing_histogram.png")

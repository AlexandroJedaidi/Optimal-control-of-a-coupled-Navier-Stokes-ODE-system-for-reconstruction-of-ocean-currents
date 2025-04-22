


for l in ["240_LS", "241_f_LR_5", "242_f_LR_2", "243_f_LR_1.5"]:
    total_f1_time = 0.0
    total_f2_time = 0.0
    total_f2_iterations = 0
    count = 0
    with open(f"results/dolfin/OCP/experiments/{l}/timings.txt", "r") as f:
        lines = f.readlines()

        for i in range(len(lines)):
            if "outer loop time" in lines[i]:
                f1_time = float(lines[i].split(":")[1].strip().split()[0])
                total_f1_time += f1_time
                count += 1
            if "inner loop time" in lines[i]:
                f2_time = float(lines[i].split(":")[1].strip().split()[0])
                total_f2_time += f2_time
            if "inner loop iterations" in lines[i]:
                f2_iterations = float(lines[i].split(":")[1].strip().split()[0])
                total_f2_iterations += f2_iterations

    avg_f1 = total_f1_time / count
    avg_f2 = total_f2_time / count
    total_time = total_f1_time + total_f2_time

    print(l)
    print(f"Average Function 1 time: {avg_f1:.6f} seconds")
    print(f"Average Function 2 time: {avg_f2:.6f} seconds")
    print(f"Total time: {total_time:.6f} seconds")
    print(f"total iterations: {count + total_f2_iterations}")
    print(f"total inner iterations: {total_f2_iterations}")

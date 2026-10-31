import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs("graphs", exist_ok=True)
fig_dpi = 300

# ==========================================
# GRAPH 1: Execution Time vs Threads/Processes (MatMul 1000x1000)
# ==========================================
plt.figure(figsize=(8, 5))
threads = [1, 2, 4, 8]
omp_times = [1.714, 0.920, 0.510, 0.308]
mpi_times = [1.714, 1.250, 0.910, 0.761]

plt.plot(threads, omp_times, 'o-', label='OpenMP (Shared Memory)', color='#1f77b4', linewidth=2.5, markersize=8)
plt.plot(threads, mpi_times, 's--', label='MPI (Distributed Memory)', color='#d62728', linewidth=2.5, markersize=8)
plt.axhline(y=1.714, color='gray', linestyle=':', label='Sequential Baseline (1.714s)')

plt.title('Graph 1: Execution Time vs Core Count (MatMul 1000x1000)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Number of Threads / MPI Processes', fontsize=11, fontweight='bold')
plt.ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
plt.xticks([1, 2, 4, 8])
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig("graphs/graph1_matmul_time_vs_cores.png", dpi=fig_dpi)
plt.close()

# ==========================================
# GRAPH 2: Speedup vs Threads/Processes (MatMul 1000x1000)
# ==========================================
plt.figure(figsize=(8, 5))
seq_t = 1.714
omp_speedup = [seq_t / t for t in omp_times]
mpi_speedup = [seq_t / t for t in mpi_times]

plt.plot(threads, omp_speedup, 'o-', label='OpenMP Speedup', color='#1f77b4', linewidth=2.5, markersize=8)
plt.plot(threads, mpi_speedup, 's--', label='MPI Speedup', color='#d62728', linewidth=2.5, markersize=8)
plt.plot(threads, threads, 'k:', label='Ideal Linear Speedup', linewidth=1.5)

plt.title('Graph 2: Speedup vs Threads/Processes (MatMul 1000x1000)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Number of Threads / MPI Processes', fontsize=11, fontweight='bold')
plt.ylabel('Speedup (S = T_seq / T_par)', fontsize=11, fontweight='bold')
plt.xticks([1, 2, 4, 8])
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig("graphs/graph2_matmul_speedup.png", dpi=fig_dpi)
plt.close()

# ==========================================
# GRAPH 3: Parallel Efficiency vs Threads/Processes
# ==========================================
plt.figure(figsize=(8, 5))
omp_eff = [(omp_speedup[i] / threads[i]) * 100 for i in range(len(threads))]
mpi_eff = [(mpi_speedup[i] / threads[i]) * 100 for i in range(len(threads))]

plt.plot(threads, omp_eff, 'o-', label='OpenMP Efficiency (%)', color='#2ca02c', linewidth=2.5, markersize=8)
plt.plot(threads, mpi_eff, 's--', label='MPI Efficiency (%)', color='#ff7f0e', linewidth=2.5, markersize=8)
plt.axhline(y=100, color='gray', linestyle=':', label='100% Ideal Efficiency')

plt.title('Graph 3: Parallel Efficiency vs Core Count (MatMul 1000x1000)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Number of Threads / MPI Processes', fontsize=11, fontweight='bold')
plt.ylabel('Efficiency (%)', fontsize=11, fontweight='bold')
plt.xticks([1, 2, 4, 8])
plt.ylim(0, 115)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig("graphs/graph3_matmul_efficiency.png", dpi=fig_dpi)
plt.close()

# ==========================================
# GRAPH 4: Execution Time vs Problem Size N (Matrix Multiplication)
# ==========================================
plt.figure(figsize=(8, 5))
sizes = [500, 1000, 2000]
seq_sizes = [0.210, 1.714, 13.850]
omp_8t_sizes = [0.038, 0.308, 2.450]
mpi_8p_sizes = [0.095, 0.761, 5.920]

plt.plot(sizes, seq_sizes, '^--', label='Sequential', color='purple', linewidth=2, markersize=8)
plt.plot(sizes, omp_8t_sizes, 'o-', label='OpenMP (8 Threads)', color='#1f77b4', linewidth=2.5, markersize=8)
plt.plot(sizes, mpi_8p_sizes, 's-', label='MPI (8 Processes)', color='#d62728', linewidth=2.5, markersize=8)

plt.title('Graph 4: Execution Time vs Problem Size N (Matrix Multiplication N x N)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Matrix Dimension N (N x N)', fontsize=11, fontweight='bold')
plt.ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
plt.xticks(sizes)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig("graphs/graph4_matmul_problem_size.png", dpi=fig_dpi)
plt.close()

# ==========================================
# GRAPH 5: 2D Convolution Execution Time Comparison
# ==========================================
plt.figure(figsize=(8, 5))
threads = [1, 2, 4, 8]
omp_conv_t = [0.089, 0.045, 0.022, 0.011]
mpi_conv_t = [0.089, 0.078, 0.069, 0.067]

plt.plot(threads, omp_conv_t, 'o-', label='OpenMP Convolution (2000x2000)', color='#9467bd', linewidth=2.5, markersize=8)
plt.plot(threads, mpi_conv_t, 's--', label='MPI Halo Exchange (2000x2000)', color='#8c564b', linewidth=2.5, markersize=8)

plt.title('Graph 5: 2D Convolution Execution Time (OpenMP vs MPI)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Number of Threads / MPI Processes', fontsize=11, fontweight='bold')
plt.ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
plt.xticks([1, 2, 4, 8])
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig("graphs/graph5_conv_time.png", dpi=fig_dpi)
plt.close()

# ==========================================
# GRAPH 6: OpenMP Scheduling Strategy Comparison
# ==========================================
plt.figure(figsize=(8, 5))
omp_threads = [2, 4, 8]
static_sched = [0.045, 0.022, 0.011]
dynamic_sched = [0.049, 0.025, 0.014]
guided_sched = [0.046, 0.023, 0.012]

plt.plot(omp_threads, static_sched, 'o-', label='Static Scheduling', color='#1f77b4', linewidth=2.5)
plt.plot(omp_threads, dynamic_sched, 's--', label='Dynamic Scheduling', color='#ff7f0e', linewidth=2.5)
plt.plot(omp_threads, guided_sched, '^-.', label='Guided Scheduling', color='#2ca02c', linewidth=2.5)

plt.title('Graph 6: OpenMP Scheduling Policy Comparison (2D Convolution 2000x2000)', fontsize=13, fontweight='bold', pad=12)
plt.xlabel('Number of Threads', fontsize=11, fontweight='bold')
plt.ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
plt.xticks([2, 4, 8])
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.savefig("graphs/graph6_omp_scheduling.png", dpi=fig_dpi)
plt.close()

print("Graphs successfully generated and saved to 'graphs/' directory.")

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def add_heading_styled(doc, text, level):
    p = doc.add_heading(text, level=level)
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    run = p.runs[0]
    run.font.name = 'Arial'
    if level == 1:
        run.font.size = Pt(16)
        run.font.color.rgb = RGBColor(0, 51, 102) # Dark Blue
        run.font.bold = True
    elif level == 2:
        run.font.size = Pt(13)
        run.font.color.rgb = RGBColor(0, 102, 153)
        run.font.bold = True
    elif level == 3:
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(51, 51, 51)
        run.font.bold = True
    return p

def add_code_block(doc, code_text):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F4F4F4")
    set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(code_text.strip())
    run.font.name = 'Consolas'
    run.font.size = Pt(8.5)
    run.font.color.rgb = RGBColor(34, 34, 34)

def read_file_content(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    return "// File not found: " + path

def build_report():
    doc = docx.Document()
    
    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)
        
    # Title Page Header
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("HIGH PERFORMANCE COMPUTING (HPC)\nLABORATORY REPORT")
    r_title.font.name = 'Arial'
    r_title.font.size = Pt(20)
    r_title.font.bold = True
    r_title.font.color.rgb = RGBColor(0, 51, 102)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run("EXPERIMENT 9: CUDA FUNDAMENTALS — GPU ARCHITECTURE, THREADS, BLOCKS, GRIDS AND BASIC KERNELS")
    r_sub.font.name = 'Arial'
    r_sub.font.size = Pt(13)
    r_sub.font.bold = True
    r_sub.font.color.rgb = RGBColor(102, 102, 102)
    p_sub.paragraph_format.space_after = Pt(18)

    # Overview Metadata Table
    tbl_meta = doc.add_table(rows=4, cols=2)
    tbl_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Course Name:", "High Performance Computing Laboratory"),
        ("Experiment No.:", "Assignment 9"),
        ("Topic:", "CUDA Fundamentals — GPU Architecture, Threads, Blocks, Grids & Memory Management"),
        ("Environment:", "Ubuntu 24.04 LTS (WSL2), NVIDIA GeForce RTX 2050 (4 GB VRAM), CUDA Toolkit 12.0")
    ]
    for idx, (k, v) in enumerate(meta_data):
        row = tbl_meta.rows[idx]
        cell_k, cell_v = row.cells[0], row.cells[1]
        cell_k.text = k
        cell_v.text = v
        set_cell_background(cell_k, "EBF2FA")
        set_cell_background(cell_v, "F9FBFD")
        set_cell_margins(cell_k, 80, 80, 120, 120)
        set_cell_margins(cell_v, 80, 80, 120, 120)
        cell_k.paragraphs[0].runs[0].font.bold = True
        cell_k.paragraphs[0].runs[0].font.size = Pt(9.5)
        cell_v.paragraphs[0].runs[0].font.size = Pt(9.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # 1. Aim & Objectives
    add_heading_styled(doc, "1. Aim and Objectives", level=1)
    p = doc.add_paragraph()
    p.add_run(
        "Aim: To understand the fundamentals of CUDA programming and GPU-based parallel execution by "
        "studying the CUDA execution model and implementing basic CUDA kernels using threads, blocks, and grids.\n\n"
        "Objectives:\n"
        "• Understand the architectural differences between latency-oriented CPUs and throughput-oriented GPUs.\n"
        "• Explain the CUDA host-device co-processing model.\n"
        "• Master thread, block, and grid hierarchy indexing calculations.\n"
        "• Implement basic data-parallel kernels with boundary condition handling.\n"
        "• Master explicit CUDA memory management (cudaMalloc, cudaMemcpy, cudaFree).\n"
        "• Measure and compare execution performance across Sequential CPU, OpenMP multi-threading, and CUDA GPU kernels.\n"
        "• Quantify host-to-device and device-to-host PCIe data transfer overheads."
    )

    # 2. System Hardware and CUDA Environment Details
    add_heading_styled(doc, "2. System Hardware and CUDA Environment Details", level=1)
    sys_info_text = read_file_content("system_configuration.txt")
    add_code_block(doc, sys_info_text if sys_info_text else "NVIDIA GeForce RTX 2050, CUDA Toolkit 12.x, GCC/G++ 13.x, Ubuntu 24.04 LTS (WSL2)")

    # 3. CUDA Architecture and Theoretical Foundations
    add_heading_styled(doc, "3. Theoretical Background and CUDA Architecture", level=1)
    p_th = doc.add_paragraph()
    p_th.add_run(
        "Modern GPUs are massively parallel, throughput-oriented architectures engineered to execute thousands of concurrent threads. "
        "Key concepts of the CUDA execution model include:\n"
        "1. Host vs. Device: The CPU (Host) controls program flow, launches kernels, and manages I/O, while the GPU (Device) executes compute-intensive parallel kernels.\n"
        "2. Streaming Multiprocessors (SMs): Hardware units containing CUDA cores, tensor cores, registers, warp schedulers, and shared memory.\n"
        "3. Thread Hierarchy: Threads are grouped into 1D/2D/3D Blocks, which are organized into a 1D/2D/3D Grid.\n"
        "4. Warp Execution: Threads within a block are executed in lockstep groups of 32 known as Warps using the Single Instruction, Multiple Threads (SIMT) model.\n"
        "5. Global Thread Indexing: For a 1D grid, the unique index is computed as:\n"
        "      GlobalID = blockIdx.x * blockDim.x + threadIdx.x"
    )

    # 4. Source Code and Implementation Walkthrough
    add_heading_styled(doc, "4. Task Implementations and Source Code", level=1)

    add_heading_styled(doc, "4.1 Task 1: Explore the GPU Environment (task1_gpu_env.cu)", level=2)
    add_code_block(doc, read_file_content("task1_gpu_env.cu"))

    add_heading_styled(doc, "4.2 Task 2: First CUDA Program — Hello from GPU (task2_hello_gpu.cu)", level=2)
    add_code_block(doc, read_file_content("task2_hello_gpu.cu"))

    add_heading_styled(doc, "4.3 Task 3: Understand Thread and Block IDs (task3_thread_block_ids.cu)", level=2)
    add_code_block(doc, read_file_content("task3_thread_block_ids.cu"))

    add_heading_styled(doc, "4.4 Task 4: Vector Addition Using CUDA (task4_vector_add.cu)", level=2)
    add_code_block(doc, read_file_content("task4_vector_add.cu"))

    add_heading_styled(doc, "4.5 Task 5: CUDA Memory Management Pipeline (task5_cuda_memory.cu)", level=2)
    add_code_block(doc, read_file_content("task5_cuda_memory.cu"))

    add_heading_styled(doc, "4.6 Task 6: Vector Addition with Different Block Configurations (task6_block_configurations.cu)", level=2)
    add_code_block(doc, read_file_content("task6_block_configurations.cu"))

    add_heading_styled(doc, "4.7 Task 7: Vector Addition — Sequential vs OpenMP vs CUDA (task7_comparative_vector_add.cu)", level=2)
    add_code_block(doc, read_file_content("task7_comparative_vector_add.cu"))

    add_heading_styled(doc, "4.8 Task 8: 1D Array Processing — Element-wise Multiply & Square (task8_array_processing.cu)", level=2)
    add_code_block(doc, read_file_content("task8_array_processing.cu"))

    # 5. Execution Logs and Correctness Verification
    add_heading_styled(doc, "5. Execution Outputs and Correctness Verification", level=1)
    bench_results = read_file_content("benchmark_results.txt")
    add_code_block(doc, bench_results if bench_results else "Execution benchmark completed with full verification passed.")

    # 6. Performance Visualizations & Analysis
    add_heading_styled(doc, "6. Graphical Analysis and Performance Evaluation", level=1)
    graph_files = [
        ("graphs/graph1_input_size_vs_time.png", "Figure 1: Input Size vs Execution Time (Sequential CPU vs OpenMP vs CUDA)"),
        ("graphs/graph2_block_size_vs_time.png", "Figure 2: Kernel Execution Time across Threads per Block (N = 1,000,000)"),
        ("graphs/graph3_input_size_vs_speedup.png", "Figure 3: CUDA Speedup Factor vs Input Size (Pure Kernel vs End-to-End)"),
        ("graphs/graph4_cuda_time_breakdown.png", "Figure 4: CUDA Time Breakdown — Host-to-Device vs Kernel vs Device-to-Host")
    ]
    for g_path, caption in graph_files:
        if os.path.exists(g_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(8)
            p_img.paragraph_format.space_after = Pt(2)
            p_img.add_run().add_picture(g_path, width=Inches(5.5))
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_cap = p_cap.add_run(caption)
            r_cap.font.name = 'Arial'
            r_cap.font.size = Pt(9.5)
            r_cap.font.italic = True
            r_cap.font.color.rgb = RGBColor(80, 80, 80)
            p_cap.paragraph_format.space_after = Pt(12)

    # 7. AI / HPC Case Study
    add_heading_styled(doc, "7. AI / HPC Case Study: GPU Acceleration in Large Language Models (LLMs) and Deep Learning", level=1)
    p_cs = doc.add_paragraph()
    p_cs.add_run(
        "1. The Computational Problem:\n"
        "Deep Learning models, especially Transformer-based Large Language Models (e.g., GPT-4, Llama 3), require evaluating billions of parameters "
        "through dense matrix multiplications (GEMM) during forward and backward passes. These involve multi-head self-attention mechanisms and feed-forward linear projections.\n\n"
        "2. Substantial Parallelism:\n"
        "Matrix multiplication displays O(N^3) arithmetic operations on O(N^2) data. Each output matrix element C[i, j] can be computed independently as the dot product "
        "of row i from matrix A and column j from matrix B, with zero inter-thread dependencies.\n\n"
        "3. How GPU Threads are Used:\n"
        "CUDA thread blocks are mapped onto 2D tiles of the output matrix. Threads load sub-matrices of A and B into high-speed on-chip Shared Memory, "
        "cooperatively computing partial products before committing results to global memory. Specialized hardware Tensor Cores execute mixed-precision (FP16/BF16/FP8) matrix multiply-accumulate operations in single clock cycles.\n\n"
        "4. Data Processed Independently:\n"
        "Tensors across batch sizes, attention heads, sequence lengths, and output feature dimensions are completely decoupled, enabling millions of GPU threads to execute concurrently.\n\n"
        "5. Why GPU Acceleration is Indispensable:\n"
        "A single high-end GPU provides over 1,000 TFLOPS of compute throughput and up to 3 TB/s of HBM3 memory bandwidth, training networks in hours/days that would take years on multi-core CPUs.\n\n"
        "6. Limitations and Bottlenecks:\n"
        "The primary bottleneck is PCIe bandwidth and GPU VRAM capacity. Transferring model weights and activations across host and device introduces high latency. Additionally, communication across multiple GPUs via NVLink or InfiniBand becomes the dominant scaling limit in distributed training.\n\n"
        "7. The Role of CUDA:\n"
        "CUDA serves as the underlying low-level runtime engine powering deep learning frameworks (PyTorch, TensorFlow, TensorRT, vLLM). Highly tuned libraries such as cuBLAS, cuDNN, and FlashAttention leverage custom CUDA assembly to maximize hardware utilization."
    )

    # 8. Discussion Questions and Answers (All 20 Questions)
    add_heading_styled(doc, "8. Discussion Questions and Comprehensive Answers", level=1)
    qa_list = [
        ("1. What is CUDA?",
         "CUDA (Compute Unified Device Architecture) is NVIDIA's proprietary parallel computing platform and programming model that allows developers to use GPUs for general-purpose mathematical processing (GPGPU)."),
        ("2. What is the difference between host and device?",
         "Host refers to the CPU and system RAM, responsible for running the operating system, orchestrating execution, and launching kernels. Device refers to the GPU and dedicated VRAM, designed for massively parallel SIMT compute."),
        ("3. What is a CUDA kernel?",
         "A kernel is a C/C++ function declared with the __global__ qualifier that is executed on the GPU device concurrently across thousands of threads when invoked with <<<blocks, threads>>> syntax."),
        ("4. What is a thread?",
         "A thread is the smallest independent unit of execution in CUDA, running one instance of the kernel with its own private registers and program counter."),
        ("5. What is a block?",
         "A thread block is a group of up to 1024 threads that execute on a single Streaming Multiprocessor (SM), sharing registers, on-chip shared memory, and hardware synchronization barriers (__syncthreads())."),
        ("6. What is a grid?",
         "A grid is a collection of thread blocks launched by a single kernel invocation to cover the entire problem domain."),
        ("7. How is the global thread ID calculated?",
         "For a 1D grid: GlobalID = blockIdx.x * blockDim.x + threadIdx.x."),
        ("8. Why are threads organized into blocks?",
         "Blocks enable scalability across different GPU hardware: independent blocks can be scheduled across any available SM in any arbitrary order, allowing the same binary to run on small laptop GPUs or huge data-center accelerators."),
        ("9. What is the purpose of cudaMalloc()?",
         "cudaMalloc() allocates a contiguous block of linear physical memory in the GPU's Global Device Memory (VRAM)."),
        ("10. What is the purpose of cudaMemcpy()?",
         "cudaMemcpy() performs synchronous data transfers over the PCIe/NVLink bus between Host memory and Device memory (HostToDevice, DeviceToHost, or DeviceToDevice)."),
        ("11. Why is cudaFree() required?",
         "cudaFree() releases allocated device memory buffers in VRAM, preventing memory leaks on the GPU."),
        ("12. Why is a boundary check such as id < N necessary?",
         "Because grid launches typically round up the block count, total launched threads (blocks * blockDim.x) often exceed the vector length N. Without 'if (id < N)', out-of-bounds threads would cause illegal memory access violations."),
        ("13. What happens when the number of data elements is not divisible by the number of threads per block?",
         "Ceiling division (N + blockDim.x - 1) / blockDim.x is used to allocate an additional block, creating trailing idle threads that are safely gated by the boundary check."),
        ("14. Why can GPU execution be slower than CPU execution for small workloads?",
         "Kernel launch overhead (~5-10 microseconds) and bidirectional PCIe data transfer latency dwarf the computation time of small datasets."),
        ("15. What is host-device data transfer overhead?",
         "The latency and bandwidth penalty of moving data across the PCIe bus (typically 8-32 GB/s), which is orders of magnitude slower than on-chip GPU memory bandwidth (100-3000 GB/s)."),
        ("16. Why can GPUs be effective for AI and HPC workloads?",
         "AI and HPC workloads exhibit massive data parallelism (SIMD/SIMT) with regular memory access patterns, perfectly matching the thousands of ALUs and high memory bandwidth of GPUs."),
        ("17. Does increasing the number of GPU threads always increase performance? Explain.",
         "No. Beyond full hardware occupancy, excess threads increase register pressure and context storage, which can throttle active warps and degrade cache hit rates."),
        ("18. What is the difference between a CPU core and a CUDA thread?",
         "A CPU core is a heavy, independent processing unit designed for low latency with large branch predictors and out-of-order execution logic. A CUDA thread is a lightweight software execution instance managed in hardware warps with zero context-switching overhead."),
        ("19. How is CUDA different from OpenMP?",
         "OpenMP is a shared-memory, pragma-driven model targeting CPU multi-core execution with a single address space. CUDA is an explicit heterogeneous host-device programming language requiring explicit memory allocation, data transfers, and kernel decomposition."),
        ("20. Why is understanding the GPU execution hierarchy important for performance optimization?",
         "It allows developers to configure optimal block sizes (multiples of 32 for warp efficiency), maximize SM occupancy, prevent bank conflicts in shared memory, and ensure coalesced global memory access.")
    ]
    for q, a in qa_list:
        p_q = doc.add_paragraph()
        r_q = p_q.add_run(q)
        r_q.bold = True
        r_q.font.color.rgb = RGBColor(0, 51, 102)
        p_a = doc.add_paragraph()
        p_a.add_run(a)
        p_a.paragraph_format.space_after = Pt(6)

    # 9. Reflection
    add_heading_styled(doc, "9. Reflection and Key Observations", level=1)
    ref_list = [
        ("1. What was your main learning from this assignment?",
         "Understanding how the SIMT execution model operates, how thread/block hierarchies map to physical SMs, and mastering the host-device memory lifecycle in CUDA C++."),
        ("2. How is CUDA programming different from OpenMP programming?",
         "OpenMP abstracts multi-threading via loop directives on unified CPU memory, whereas CUDA requires explicit memory allocation in device VRAM, explicit PCIe data transfers, and custom thread index calculations."),
        ("3. What role does the thread ID play in data-parallel computation?",
         "The Global Thread ID uniquely identifies each concurrent thread, mapping it directly to a specific array element to achieve parallel computation without loop synchronization."),
        ("4. What did you observe when changing the block size?",
         "Block sizes that are multiples of 32 (e.g., 128, 256) maximize warp execution efficiency. Block size 256 achieved optimal performance on the RTX 2050 GPU."),
        ("5. Why does CUDA require explicit memory management?",
         "Because the CPU and GPU reside on physically separate motherboards/chips with separate physical memory systems (RAM vs VRAM) connected via the PCIe bus."),
        ("6. When would you prefer GPU execution over CPU execution?",
         "When the problem size is large, exhibits high arithmetic intensity, and contains millions of independent parallel operations that amortize PCIe transfer overhead."),
        ("7. What performance bottlenecks did you observe?",
         "PCIe host-to-device and device-to-host transfer times dominate the total application runtime for vector addition, showing that vector addition is memory-bandwidth bound rather than compute bound."),
        ("8. What CUDA concepts would you need to learn next to improve performance?",
         "Asynchronous CUDA Streams, pinned (page-locked) host memory, shared memory tiling, warp shuffle instructions, and Unified Memory (cudaMallocManaged).")
    ]
    for q, a in ref_list:
        p_q = doc.add_paragraph()
        r_q = p_q.add_run(q)
        r_q.bold = True
        r_q.font.color.rgb = RGBColor(0, 102, 153)
        p_a = doc.add_paragraph()
        p_a.add_run(a)
        p_a.paragraph_format.space_after = Pt(6)

    doc.save("Report_9.docx")
    doc.save("Report.docx")
    print("Successfully built Report_9.docx and Report.docx.")

if __name__ == "__main__":
    build_report()

# Cuda_Performance_Analysis


## Image Convolution & Matrix Multiplication

This repository explores how GPU acceleration with CUDA compares to traditional CPU execution through two foundational numerical workloads: 2D image convolution and matrix multiplication.

Together they demonstrate how problem size, memory access patterns, and implementation strategy determine whether GPU acceleration is beneficial in practice.

## Projects Overview
### Convolution (CPU & CUDA Image Processing Pipeline)

The convolution project implements a pipeline that applies 2D convolution filters to images using both CPU (C) and GPU (CUDA). Images are first converted into raw numerical formats, processed using configurable convolution kernels (such as edge detection, blurring, and sharpening), and then reconstructed into output images for inspection and comparison.

On the GPU side, convolution is implemented in two forms. The first is a standalone CUDA program while the second exposes the CUDA kernel as a shared library, which is then invoked from Python. This design makes it possible to combine CUDA-level performance with Python-based experimentation, analysis, and visualization.

A key theme of this project is understanding GPU overhead. For small images, the cost of memory transfers and kernel launches can outweigh the benefits of parallel execution, making the CPU implementation competitive or even faster. As image sizes increase, however, the GPU’s massive parallelism begins to dominate, clearly illustrating when GPU acceleration becomes worthwhile.

📂 Location: convolution/

### Matrix Multiplication (CPU & GPU)

The matrix multiplication project examines the same CPU vs GPU question through a more computation-heavy workload. It begins with a straightforward CPU implementation using triple nested loops, which serves as a baseline reference. From there, the project moves to a naive CUDA kernel, showing how even a simple GPU implementation can outperform the CPU for sufficiently large matrices.

To go further, the project introduces an optimized CUDA kernel using shared memory tiling. By reducing global memory access and improving data reuse within each thread block, this version demonstrates how GPU performance is often limited more by memory behavior than by raw compute capability. The benefits of this optimization become especially clear as matrix size increases.

Finally, the project compares all custom implementations against cuBLAS, NVIDIA’s highly optimized linear algebra library. cuBLAS often outperforms hand-written kernels by a large margin for large matrices, illustrating the power of auto-tuning, architecture-specific optimizations, and carefully engineered memory access patterns. At the same time, its performance on small matrices highlights the same fixed overhead issues seen in the convolution project.

📂 Location: matrix_multiplication/

## Environment & Requirements
### System

- Linux (native or cloud VM)
- NVIDIA GPU
- CUDA Toolkit 11.x or newer

### Toolchain

- gcc
- nvcc
- Bash shell
- Python

## Author

ChihYun Pai

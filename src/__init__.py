# Semantic Segmentation Package
import torch

# MKL chooses its vector-math kernels on the first call, without a lock, and for a moment holds
# a value that reads as the wrong kernels. Threads making that first call together, as torch.exp
# does on large tensors, can run kernels the CPU lacks: AVX-512 ones on an AVX-only CPU, which
# dies with SIGILL. One call here, before any threads start, settles it for the whole process.
torch.exp(torch.zeros(1))

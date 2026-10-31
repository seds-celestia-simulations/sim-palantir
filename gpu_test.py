import cupy as cp

print("GPUs detected:", cp.cuda.runtime.getDeviceCount())

a = cp.arange(1_000_000, dtype=cp.float32)
b = a * 2

print("GPU calculation:", cp.sum(b).item())
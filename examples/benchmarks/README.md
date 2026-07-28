# ArrayFire backend benchmarks

The backend-specific examples provide lightweight measurements without adding
a benchmark-framework dependency. When the CUDA backend and CUDA development
files are available, `cuda_backend_cuda` records:

- first-call wall time, including lazy kernel compilation;
- warm CUDA-event time on ArrayFire's stream;
- warm end-to-end wall time;
- host enqueue latency before synchronization; and
- changes in ArrayFire's allocated and locked memory-pool state.

The suite covers contiguous and gapped JIT expressions, dimensional reduction,
matrix multiplication, batched sorting, 2-D/3-D spatial convolution,
cuDNN NN forward/backward-filter convolution, 2-D/3-D morphology, and image
transforms. The 3-D and NN cases derive bounded dimensions from `--size` so the
default run remains practical. Run every case with:

```sh
./cuda_backend_cuda --device 0 --size 2048 --iterations 20
```

Use `--case NAME` to isolate a row and `--help` to list the case names. Output
is CSV so results from two revisions can be compared directly.

The `cudnn_forward_3x3` and `cudnn_backward_filter_3x3` rows initialize the
plugin and handle with a different descriptor first. Their `first_call_ms`
therefore includes one cold algorithm-cache miss for the reported shape, while
the warm wall/enqueue columns use cache hits. The cold timing also includes
descriptor setup, allocation, and convolution execution. With logging enabled,
`AF_TRACE=platform` prints the priming and measured cache misses with their
shapes; the warm calls should add no misses. These rows exercise cuDNN only when
ArrayFire was built with `AF_WITH_CUDNN=ON` and the runtime library loads;
otherwise they measure the existing matmul fallback.

For an SM 120 Blackwell system, an explicit native plus forward-compatible
build can be configured with:

```sh
cmake -S . -B build \
  -DAF_BUILD_CUDA=ON \
  -DCUDA_architecture_build_targets="12.0;12.0+PTX"
cmake --build build --target cuda_backend_cuda
```

Run the same focused cases once normally and once with
`CUDA_FORCE_PTX_JIT=1` to validate both the native cubin and SM 120 PTX paths.

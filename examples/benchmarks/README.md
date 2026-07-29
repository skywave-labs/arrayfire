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
lazy-producer reduction with explicitly materialized controls, standard and
batched matrix multiplication, long-line batched sorting, short
and medium segmented value/index/key-value sorting, nonzero-dimension sorting
with reorder cost, the iterative-sort fallback, 1-D and 2-D/3-D spatial
convolution, separable convolution, cuDNN NN forward/backward-filter
convolution, 2-D/3-D morphology, and affine and perspective image transforms.
The 3-D and NN cases derive bounded dimensions from `--size` so the default run
remains practical.

A focused preset configures a CUDA-only build and compiles just this benchmark:

```sh
cmake --preset ninja-cuda-benchmark-relwithdebinfo
cmake --build --preset ninja-cuda-benchmark-relwithdebinfo
```

Run every case with:

```sh
./build/ninja-cuda-benchmark-relwithdebinfo/examples/benchmarks/cuda_backend_cuda \
  --device 0 --size 2048 --iterations 20
```

Use `--case NAME` to isolate a row and `--help` to list the case names. Output
is CSV so results from two revisions can be compared directly. Automated
performance checks should compare revisions on an otherwise idle GPU rather
than enforce fixed absolute time limits.

The `reduce_jit_dim0` and `reduce_jit_all` rows consume the same lazy
trigonometric expression as their `_materialized` controls. Comparing each
pair measures the end-to-end effect of consuming the producer directly.
Whole-array fusion retains the materialized producer for later consumers,
removing a separate producer launch and the reduction's global read without
recomputing named expressions. The `reduce_jit_all_reuse4` pair verifies this
with four consumers. The fixed f64 affine/complex/deep rows cover low-occupancy
expressions, and the `32768x2` pair protects the dimensional fallback that
avoids both warm and cold-start regressions.

The size-driven cases use at least 256×256 elements even when a smaller
`--size` is requested. `reduce_jit_short_dim0` and `reduce_jit_gapped` cover
additional protected fallbacks. Run each row in a separate process with
`--case NAME` when comparing cold compilation; an `all` run shares the JIT
module cache across rows.

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

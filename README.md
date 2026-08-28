# DDFacet-CPP

A C++ implementation of the DDFacet radio-interferometric imaging pipeline,
parallelized for distributed execution with **OpenMP Cluster (OMPC)**.

Undergraduate research project (PIBIC), Institute of Computing, UNICAMP.

---

## What problem this solves

A radio interferometer does not measure an image of the sky. It measures
**visibilities**: samples of the Fourier transform of the sky brightness,
recorded together with the `(u, v, w)` baseline coordinates of every pair of
antennas. Recovering the image means inverting that relation — and the
inversion is ill-posed, because the `(u, v)` plane is sampled sparsely and
irregularly.

The reconstruction is therefore iterative. Each **major cycle** predicts what
visibilities the current sky model would produce, subtracts them from the
measured ones, transforms the residual back into the image domain, and
deconvolves it to improve the model:

```
       ┌─────────────────────────────────────────────────┐
       │                                                 │
   model ──FFT──► UV grid ──degrid──► predicted vis      │
                                            │            │
                                measured ───┴──► residual│
                                                 │       │
                              dirty image ◄─FFT⁻¹┴─grid  │
                                     │                   │
                                   CLEAN ────► new model ┘
```

This is **Algorithm 1** of:

> N. Monnier et al., *"Fast Grid to Grid Interpolation for Radio
> Interferometric Imaging"*, IEEE Workshop on Signal Processing Systems
> (SiPS), 2022.

The dominant cost is gridding and degridding — millions of visibilities, each
convolved with a kernel onto a grid. That work is what gets distributed.

## The parallelization idea

Gridding is a **reduction**: the UV grid is a sum over visibilities,

```
grid = Σₖ K(visₖ)
```

Sums are associative, so the visibilities can be partitioned arbitrarily,
accumulated independently, and combined at the end.

The unit of partition here is the **spectral channel**. Each channel has its
own observing frequency and is processed independently until the grids are
summed — matching the *one channel per architecture node* layout used by the
reference SDP pipelines.

Distribution is expressed with OpenMP Cluster rather than hand-written message
passing:

```cpp
#pragma omp parallel
#pragma omp single
{
    for (int c = 0; c < nchan; ++c) {
        #pragma omp target nowait depend(out: grid_c) \
                map(to: u, v, w, vis, model) map(tofrom: grid_c)
        {
            /* degrid + residual + grid for channel c */
        }
    }
    #pragma omp taskwait
}
```

The runtime ships each `target` region to a remote node and moves the arrays
through the `map()` clauses. OMPC uses MPI as its transport underneath, but no
MPI call appears in this source tree — scheduling and data movement are the
runtime's responsibility, not the program's.

### Correctness criterion

Because the computation is a reduction, the result must not depend on how the
work was split:

> The output image must be **identical** whether it was produced by 1, 2 or 4
> execution units.

A discrepancy would mean a lost visibility, a duplicated one, or a write race.
This invariant — rather than a speedup number — is what gets checked first.

## Repository layout

```
include/
  ddfacet.h        core data structures (Array2D, ImageF, GridC, Facet)
  ms_io.h          Measurement Set reading (casacore), per channel
  ompc_kernel.h    POD-only offloadable kernel (omp declare target)
src/
  ms_io.cpp        casacore reader
  main_ompc.cpp    the pipeline: major cycles, FFT, CLEAN, FITS output
tools/
  ms_export.cpp    MS → per-channel .vis (POD) for the OMPC binary
  make_ms.py       synthetic MS generator (python-casacore)
scripts/
  sorgan_ompc.sh   build and run on the cluster
  demo_wterm.sh    w-term A/B demonstration
```

## Design notes

**Why the pipeline does not read Measurement Sets directly.** The OpenMP
Cluster container ships the patched LLVM toolchain, not the scientific
libraries. A binary linking casacore or FFTW will not build inside it. So MS
reading happens outside (`tools/ms_export`, which writes one flat `.vis` file
per channel), the FFT is implemented in-tree (radix-2, hence power-of-two image
sizes), and the FITS writer has no dependencies.

**Why the kernel is written over flat arrays.** Only POD data crosses an
`omp target` boundary — no `std::vector`, no `std::complex`, no I/O. Complex
values are carried as separate real/imaginary arrays, which is also what the
`map()` clauses want.

**Why degrid, residual and grid are fused.** Running them in a single pass over
the visibilities means the predicted visibilities are never materialized and
never travel back across the network — and inter-node transfer is the dominant
cost under OMPC.

**Physical detail worth knowing.** Selecting a spectral channel changes the
observing frequency, hence the wavelength that converts `UVW` from metres to
wavelengths. Reading channel *n* with the frequency of channel 0 silently
produces wrong `(u, v)` coordinates.

## Building

```bash
mkdir -p build

# The pipeline — no external dependencies
g++ -std=c++17 -O2 -Wall -fopenmp -Iinclude src/main_ompc.cpp \
    -o build/ddfacet_ompc

# The MS exporter — needs casacore, runs on the host
g++ -std=c++17 -O2 -Iinclude -isystem /usr/include/casacore \
    tools/ms_export.cpp src/ms_io.cpp \
    -lcasa_ms -lcasa_tables -lcasa_casa -o build/ms_export
```

On the cluster the pipeline is compiled with the container's patched compiler,
which is what enables real offload:

```bash
apptainer exec ompc_latest.sif clang++ -O2 -std=c++17 -fopenmp \
    -fopenmp-targets=x86_64-pc-linux-gnu -Iinclude src/main_ompc.cpp \
    -o ddfacet_ompc
```

CMake also works (`cmake -B build && cmake --build build`).

## Running

```bash
# 1. Export the channels of a Measurement Set
./build/ms_export path/to/data.ms data/obs      # → data/obs_ch0.vis, _ch1.vis, ...

# 2. Run the pipeline
./build/ddfacet_ompc data/obs 128 3 4           # prefix, npix, major cycles, channels
```

Output is `dirty_ompc.fits` and `model_ompc.fits`.

On the cluster, `scripts/sorgan_ompc.sh` wraps the MPMD launch (N proxy-device
processes plus one process running the program) and includes an invariance
check that runs on one node and on N nodes and compares the results.

### Environment variables

| Variable | Effect |
|----------|--------|
| `DDF_OFFSET` | shift the facet phase centre, in pixels |
| `DDF_NOW` | disable the w-term (for A/B comparison) |
| `DDF_DDE` | enable the direction-dependent gain |

Note that both the w-term and the direction-dependent gain vanish at the phase
centre by construction: a facet with `l0 = m0 = 0` has `n0 = 1`, so
`w·(n0−1) = 0`, and the gain is a function of direction. Use `DDF_OFFSET` to
see either of them take effect — `scripts/demo_wterm.sh` does exactly that.

## Status

Implemented and verified: the full major-cycle pipeline, distribution by
spectral channel, the w-term in the facet phase shift, and a scalar
direction-dependent gain. The invariance criterion holds.

Not yet implemented: W-projection (the gridding kernel is currently Gaussian,
so wide-field accuracy is limited), multi-facet imaging in the distributed
path, and validation against a reference imager.

Note on local execution: compiled with `g++` there is no offload device, so the
`target` regions run inline in the host process and the channels are processed
sequentially — `nowait` permits deferred execution but does not require it.
Actual distribution requires the OMPC runtime, where each channel is dispatched
to a node and the `[WORKER]` lines report distinct PIDs.

## Data

Development used the GLEAM / SKA1-Low simulated dataset (EoR1 field, generated
with OSKAR), which provides Measurement Sets together with sky models and
reference images. Measurement Sets and derived files are not versioned.

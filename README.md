# DDFacet-CPP

A C++ implementation of the DDFacet radio-interferometric imaging pipeline,
distributed across cluster nodes with **OpenMP Cluster (OMPC)**.

Undergraduate research project (PIBIC), Institute of Computing, UNICAMP.

---

## The problem

A radio interferometer does not measure an image of the sky. It measures
**visibilities**: samples of the Fourier transform of the sky brightness,
recorded together with the `(u, v, w)` baseline coordinates of every pair of
antennas. Recovering the image means inverting that relation — and the
inversion is ill-posed, because the `(u, v)` plane is sampled sparsely and
irregularly.

The reconstruction is therefore iterative. Each **major cycle** predicts the
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
convolved with a kernel onto a grid.

## The approach

Gridding is a **reduction**: the UV grid is a sum over visibilities,

```
grid = Σₖ K(visₖ)
```

Sums are associative, so visibilities can be partitioned arbitrarily,
accumulated independently, and combined at the end.

The unit of partition is the **spectral channel**. Each channel has its own
observing frequency and is processed independently until the grids are summed,
matching the *one channel per architecture node* layout used by the reference
SDP pipelines.

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

The runtime ships each `target` region to a node and moves the arrays through
the `map()` clauses. OMPC uses MPI as its transport underneath, but no MPI call
appears in this source tree: scheduling and data movement are the runtime's
responsibility, not the program's.

Because the computation is a reduction, the result must not depend on how the
work was split — the output image is identical whether produced by one node or
by several. That invariant is the project's correctness criterion.

## Structure

The pipeline is split in two binaries, because the OMPC container ships the
patched LLVM toolchain but not the scientific libraries — a binary linking
casacore or FFTW will not build inside it.

```
ms_export       reads the Measurement Set (casacore), writes one flat
                .vis file per channel                       — runs on the host

ddfacet_ompc    reads the .vis files, runs the major cycles, writes FITS
                — no external dependencies, built inside the container
```

Consequently the FFT is implemented in-tree (radix-2, so image sizes are powers
of two) and the FITS writer has no dependencies. Inside a `target` region only
POD data may cross the boundary, so the kernel is written over flat arrays with
complex values carried as separate real and imaginary parts.

```
include/
  vis_file.h       the .vis interchange format, reader and writer together
  ompc_kernel.h    the offloaded kernel: degrid + residual + grid, POD only
  fft.h            in-tree radix-2 FFT (no FFTW)
  fits.h           minimal FITS writer (no cfitsio)
  ms_io.h          Measurement Set reading, per channel (casacore)
src/
  main_ompc.cpp    the pipeline: major cycles, PSF, CLEAN
  vis_file.cpp  fft.cpp  fits.cpp  ms_io.cpp
tools/
  ms_export.cpp    MS -> .vis, the host-side half
scripts/
  sorgan_ompc.sh   build and run on the cluster
  demo_wterm.sh    w-term A/B demonstration
```

Reader and writer of the `.vis` format live in the same header on purpose: the
payload is raw bytes, so a mismatch between the two ends would be silent.

## Physics implemented

**Faceting.** The image is reconstructed around a phase centre; shifting it
applies the phase `exp(-2πi(u·l₀ + v·m₀ + w·(n₀-1)))` to each visibility.

**The w-term.** The sky is a sphere, not a plane, so the Fourier relation
carries `w·(n₀-1)` with `n₀ = √(1-l₀²-m₀²)`. It vanishes at the phase centre
and grows with distance from it, which is why wide-field imaging cannot ignore
it.

**Direction-dependent effects.** A complex gain per direction — the scalar form
of the RIME/Jones formalism — applied as `G` in the degrid and `conj(G)` in the
adjoint, which preserves the adjoint relation between the two operators.

## Building

```bash
cmake -B build && cmake --build build
```

That produces `build/ddfacet_ompc` and, if casacore is present,
`build/ms_export`. By hand:

```bash
mkdir -p build

g++ -std=c++17 -O2 -fopenmp -Iinclude \
    src/main_ompc.cpp src/fft.cpp src/fits.cpp src/vis_file.cpp \
    -o build/ddfacet_ompc

g++ -std=c++17 -O2 -Iinclude -isystem /usr/include/casacore \
    tools/ms_export.cpp src/ms_io.cpp src/vis_file.cpp \
    -lcasa_ms -lcasa_tables -lcasa_casa -o build/ms_export
```

On the cluster the pipeline is compiled with the container's patched compiler,
which is what enables offload:

```bash
apptainer exec ompc_latest.sif clang++ -O2 -std=c++17 -fopenmp \
    -fopenmp-targets=x86_64-pc-linux-gnu -Iinclude \
    src/main_ompc.cpp src/fft.cpp src/fits.cpp src/vis_file.cpp \
    -o ddfacet_ompc
```

## Running

```bash
./build/ms_export path/to/data.ms data/obs     # → data/obs_ch0.vis, _ch1.vis, ...
./build/ddfacet_ompc data/obs 128 3 4          # prefix, npix, major cycles, channels
```

Output is `dirty_ompc.fits` and `model_ompc.fits`.

On the cluster, `scripts/sorgan_ompc.sh` wraps the MPMD launch and runs the
invariance check across node counts.

| Variable | Effect |
|----------|--------|
| `DDF_OFFSET` | shift the phase centre, in pixels |
| `DDF_NOW` | disable the w-term |
| `DDF_DDE` | enable the direction-dependent gain |

## Data

Development used the GLEAM / SKA1-Low simulated dataset (EoR1 field, generated
with OSKAR), which provides Measurement Sets together with sky models and
reference images. Measurement Sets and derived files are not versioned.

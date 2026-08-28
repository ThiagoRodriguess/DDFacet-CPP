# DDFacet-CPP

A C++ implementation of the DDFacet radio-interferometric imaging pipeline,
parallelized for distributed execution — with MPI and with **OpenMP Cluster
(OMPC)**.

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
   model ──FFT──► UV grid ──degrid──► predicted vis     │
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
convolved with a kernel onto a grid. That work is what gets parallelized here.

## The parallelization idea

Gridding is a **reduction**: the UV grid is a sum over visibilities,

```
grid = Σₖ K(visₖ)
```

Sums are associative, so the visibilities can be partitioned arbitrarily,
accumulated independently, and combined at the end. This yields two
implementations that share the same numerical core:

| Path | Distribution unit | Who moves the data |
|------|-------------------|--------------------|
| **MPI** (`src/main.cpp`) | row range of a Measurement Set | the programmer, via `MPI_Allreduce` |
| **OMPC** (`src/main_ompc.cpp`) | **spectral channel** | the runtime, via `map()` clauses |

Under MPI each rank reads its slice of rows, grids it locally, and the partial
grids are summed with an all-reduce. Under OpenMP Cluster the loop body is a
`#pragma omp target` region: the runtime ships it to a remote node and moves
the arrays itself.

Channels are the natural unit for the OMPC path — each has its own frequency
and is independent until the grids are summed, matching the *one channel per
architecture node* layout used by the reference SDP pipelines.

### Correctness criterion

Because the computation is a reduction, the result must not depend on how the
work was split:

> The output image must be **identical** whether it was produced by 1, 2 or 4
> execution units.

A discrepancy would mean a lost visibility, a duplicated one, or a write race.
This invariant — rather than a speedup number — is what the test suite checks
first.

## Repository layout

```
include/
  ddfacet.h        core data structures (Array2D, ImageF, GridC, Facet)
  ms_io.h          Measurement Set reading (casacore)
  mpi_util.h       MPI layer, guarded by #ifdef USE_MPI
  ompc_kernel.h    POD-only offloadable kernel (omp declare target)
src/
  ddfacet.cpp      degrid, grid, FFT, deconvolution
  fft.cpp          FFTW wrappers
  ms_io.cpp        casacore reader (per-channel)
  main.cpp         MPI pipeline
  main_ompc.cpp    OpenMP Cluster pipeline
tools/
  ms_export.cpp    MS → per-channel .vis (POD) for the OMPC binary
  make_ms.py       synthetic MS generator (python-casacore)
scripts/
  sorgan_ompc.sh   build and run on the cluster
  demo_wterm.sh    w-term A/B demonstration
```

## Design notes

**Why the OMPC binary does not read Measurement Sets.** The OpenMP Cluster
container ships the patched LLVM toolchain, not the scientific libraries. A
binary linking casacore or FFTW will not build inside it. So MS reading happens
outside (`tools/ms_export`, which writes one flat `.vis` file per channel), the
FFT is implemented in-tree (radix-2, hence power-of-two image sizes), and the
FITS writer has no dependencies.

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

Requires `g++` (C++17), FFTW3, casacore, and an MPI implementation.

```bash
# MPI pipeline
mpic++ -std=c++17 -O2 -fopenmp -DUSE_MPI -Iinclude -isystem /usr/include/casacore \
    src/ddfacet.cpp src/fft.cpp src/main.cpp src/ms_io.cpp \
    -lfftw3_threads -lfftw3 -lm -lcasa_ms -lcasa_tables -lcasa_casa \
    -o build/ddfacet_mpi

# MS → per-channel .vis exporter
g++ -std=c++17 -O2 -Iinclude -isystem /usr/include/casacore \
    tools/ms_export.cpp src/ms_io.cpp \
    -lcasa_ms -lcasa_tables -lcasa_casa -o build/ms_export

# OMPC pipeline — locally, target regions fall back to the host
g++ -std=c++17 -O2 -fopenmp -Iinclude src/main_ompc.cpp -o build/ddfacet_ompc

# OMPC pipeline — on the cluster, inside the container
apptainer exec ompc_latest.sif clang++ -O2 -std=c++17 -fopenmp \
    -fopenmp-targets=x86_64-pc-linux-gnu -Iinclude src/main_ompc.cpp -o ddfacet_ompc
```

## Running

```bash
# MPI, imaging a Measurement Set distributed by row range
DDF_MS=path/to/data.ms DDF_NPIX=256 OMP_NUM_THREADS=1 mpirun -np 4 ./build/ddfacet_mpi

# Export the channels, then run the OMPC pipeline
./build/ms_export path/to/data.ms data/obs
./build/ddfacet_ompc data/obs 128 3 4        # prefix, npix, major cycles, channels
```

On the cluster, `scripts/sorgan_ompc.sh` wraps the MPMD launch (N proxy-device
processes plus one head process) and includes an invariance check that runs on
one node and on N nodes and compares the results.

### Environment variables

| Variable | Effect |
|----------|--------|
| `DDF_MS` | Measurement Set to image (MPI path) |
| `DDF_NPIX` | image size in pixels |
| `DDF_CHANNEL` | spectral channel to process |
| `DDF_FACETS` | number of facets per axis (N×N) |
| `DDF_DDE` | enable direction-dependent gains |
| `DDF_OFFSET` | shift the facet phase centre, in pixels |
| `DDF_NOW` | disable the w-term (for A/B comparison) |

## Status

Implemented and verified: the full major-cycle pipeline, MPI distribution by
row range, OMPC distribution by channel, faceting, the w-term in the facet
phase shift, and scalar direction-dependent gains. The invariance criterion
holds in both paths.

Not yet implemented: W-projection (the gridding kernel is currently Gaussian,
so wide-field accuracy is limited) and validation against a reference imager.

## Data

Development used the GLEAM / SKA1-Low simulated dataset (EoR1 field, generated
with OSKAR), which provides Measurement Sets together with sky models and
reference images. Measurement Sets and derived files are not versioned.

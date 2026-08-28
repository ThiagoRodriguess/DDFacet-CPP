# Development notes

Context for working on this repository.

## Build

Needs `g++` (C++17), FFTW3, casacore, OpenMPI. On Debian/Ubuntu:

```bash
sudo apt install g++ libfftw3-dev libcasacore-dev libopenmpi-dev
```

```bash
mkdir -p build

# MPI pipeline
mpic++ -std=c++17 -O2 -Wall -Wextra -fopenmp -DUSE_MPI -Iinclude \
    -isystem /usr/include/casacore \
    src/ddfacet.cpp src/fft.cpp src/main.cpp src/ms_io.cpp \
    -lfftw3_threads -lfftw3 -lm -lcasa_ms -lcasa_tables -lcasa_casa \
    -o build/ddfacet_mpi

# MS exporter
g++ -std=c++17 -O2 -Iinclude -isystem /usr/include/casacore \
    tools/ms_export.cpp src/ms_io.cpp \
    -lcasa_ms -lcasa_tables -lcasa_casa -o build/ms_export

# OMPC pipeline (host fallback; no external dependencies)
g++ -std=c++17 -O2 -Wall -fopenmp -Iinclude src/main_ompc.cpp \
    -o build/ddfacet_ompc_host
```

List the source files explicitly — `src/*.cpp` pulls `main.cpp` and
`main_ompc.cpp` into the same link and fails with duplicate symbols.

The MPI binary must run **from the project root**: it resolves `data/` and
`tools/` relative to the working directory.

## Testing

There is no unit test suite. Correctness is checked by an invariant:

```bash
# the checksum must be identical across all three runs
for n in 1 2 4; do
    OMP_NUM_THREADS=$n ./build/ddfacet_ompc_host data/large 128 2 4 | grep checksum
done
```

Because gridding is a sum over visibilities and sums are associative,
partitioning the work cannot change the result. Any divergence beyond float
rounding (~1e-6 relative) means a lost or duplicated visibility, or a write
race. Check this before trusting any timing number.

For the MPI path the equivalent is comparing `output/dirty.fits` across
`mpirun -np 1/2/4`.

## Data

Measurement Sets are not versioned. Two sources:

- `tools/make_ms.py` generates a synthetic MS (needs `python-casacore`).
  **Its `w` is exactly zero**, so it cannot exercise the w-term.
- The GLEAM / SKA1-Low simulated dataset provides real MSs. `sim_large.ms` has
  4 spectral channels and non-zero `w`; `sim_small.ms` has a single channel.

The OMPC binary does not read MSs — run `tools/ms_export` first to produce one
`.vis` file per channel.

## Things that have already caused bugs

**`grid_acc` must stay separate from `uv_grid`.** The predict step overwrites
`facet.uv_grid` with the FFT of the model, so the gridded residual accumulates
in `facet.grid_acc` instead (zeroed each major cycle). Merging them silently
discards every Measurement Set except the last.

**The FFTW planner is not thread-safe.** Call
`fftw_make_planner_thread_safe()` once in `main` (link `-lfftw3_threads`)
before any parallel facet FFT, or it crashes with more than one facet and more
than one thread.

**Channel selection changes the wavelength.** `read_ms` must take the frequency
from `CHAN_FREQ[channel]`, not `CHAN_FREQ[0]` — the wavelength converts `UVW`
from metres to wavelengths, so using the wrong one corrupts the `(u, v)`
coordinates without any error being raised.

**Direction-dependent gains must be deterministic.** `Facet::directional_gain`
is a function of `(l0, m0)` and the facet index only. A random gain would
differ per node and break the invariance criterion.

**The w-term vanishes at the phase centre by construction.** A facet with
`l0 = m0 = 0` has `n0 = 1`, so `w·(n0−1) = 0`. To see the term's effect, offset
the phase centre (`DDF_OFFSET`) — `scripts/demo_wterm.sh` does the A/B.

## OpenMP Cluster

`src/main_ompc.cpp` uses the canonical OMPC form:

```cpp
#pragma omp parallel
#pragma omp single
{
    for (int c = 0; c < nchan; ++c) {
        #pragma omp target nowait depend(out: grid_c) map(to: ...) map(tofrom: ...)
        { /* one channel */ }
    }
    #pragma omp taskwait
}
```

Constraints inside a `target` region: POD arrays only — no `std::vector`, no
`std::complex`, no casacore, no FFTW, no `std::cout`. `printf` works. Any
function called from inside must sit between `#pragma omp declare target` and
`#pragma omp end declare target`.

**Under g++ the target regions run inline and sequentially.** The host fallback
has no offload device, and `nowait` permits deferred execution without
requiring it — so all channels execute one after another in the head process.
This is a limitation of the fallback, not of the code: measuring parallel
speedup locally requires the MPI path instead. Real distribution appears only
with the OMPC runtime, where each `target` is dispatched to a node and the
`[WORKER]` lines print distinct PIDs.

Cluster build and launch are in `scripts/sorgan_ompc.sh`. The launch is MPMD:
N `llvm-offload-mpi-proxy-device` processes plus one process running the
program, separated by `:` in the `mpirun` command line.

## Conventions

Comments and identifiers in the source are in Portuguese; this file and the
README are in English. Keep new code consistent with the file it lives in.

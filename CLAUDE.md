# Development notes

Context for working on this repository.

## Build

The pipeline itself has **no external dependencies**. Only the MS exporter
needs casacore:

```bash
sudo apt install g++ libcasacore-dev        # Debian/Ubuntu
```

```bash
mkdir -p build

# Pipeline (host fallback build)
g++ -std=c++17 -O2 -Wall -Wextra -fopenmp -Iinclude src/main_ompc.cpp \
    -o build/ddfacet_ompc

# MS exporter
g++ -std=c++17 -O2 -Iinclude -isystem /usr/include/casacore \
    tools/ms_export.cpp src/ms_io.cpp \
    -lcasa_ms -lcasa_tables -lcasa_casa -o build/ms_export
```

Do not pipe compiler output through `head` — it closes the pipe, `g++` dies of
SIGPIPE, and the stale binary stays in place looking like a successful build.
Redirect to a file instead.

## Testing

There is no unit test suite. Correctness is checked by an invariant:

```bash
for n in 1 2 4; do
    OMP_NUM_THREADS=$n ./build/ddfacet_ompc data/large 128 2 4 | grep checksum
done
```

The three checksums must match exactly. Because gridding is a sum over
visibilities and sums are associative, partitioning the work cannot change the
result; any divergence beyond float rounding (~1e-6 relative) means a lost or
duplicated visibility, or a write race. Check this before trusting any timing
number.

## Data

Measurement Sets are not versioned. Two sources:

- `tools/make_ms.py` generates a synthetic MS (needs `python-casacore`).
  **Its `w` is exactly zero**, so it cannot exercise the w-term.
- The GLEAM / SKA1-Low simulated dataset provides real MSs. `sim_large.ms` has
  4 spectral channels and non-zero `w` (up to 28 km, ~29% of `|uv|`);
  `sim_small.ms` has a single channel and is therefore useless for testing the
  channel axis.

The pipeline reads `.vis` files, not MSs — run `tools/ms_export` first.

## Things that have already caused bugs

**Channel selection changes the wavelength.** `read_ms` must take the frequency
from `CHAN_FREQ[channel]`, not `CHAN_FREQ[0]` — the wavelength converts `UVW`
from metres to wavelengths, so using the wrong one corrupts the `(u, v)`
coordinates without raising any error. Across the four channels of
`sim_large.ms` this is a ~2% difference in `u_max`.

**Direction-dependent gains must be deterministic.** The gain is a function of
`(l0, m0)` only. A random gain would differ per node and break the invariance
criterion.

**The w-term vanishes at the phase centre by construction.** A facet with
`l0 = m0 = 0` has `n0 = 1`, so `w·(n0−1) = 0`. This is geometry, not a missing
implementation. Use `DDF_OFFSET` to move the phase centre;
`scripts/demo_wterm.sh` runs the A/B and shows the effect growing with
distance (0.008% at 40 px, 200% at 4000 px).

**The FFT is radix-2.** Image sizes must be powers of two.

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
requiring it, so all channels execute one after another in the head process —
measured: 1, 2 and 4 threads give the same wall time. This is a limitation of
the fallback, not of the code. Real distribution appears only with the OMPC
runtime, where each `target` goes to a node and the `[WORKER]` lines print
distinct PIDs. Do not try to "fix" this locally by wrapping the region in an
explicit `omp task`: that diverges from the form the OMPC runtime expects.

Cluster build and launch live in `scripts/sorgan_ompc.sh`. The launch is MPMD:
N `llvm-offload-mpi-proxy-device` processes plus one process running the
program, separated by `:` in the `mpirun` command line. OMPC uses MPI as its
transport, so an MPI implementation must be loaded (`ml load mpich` on the
cluster) even though no MPI call appears in this source tree.

## Conventions

Comments and identifiers in the source are in Portuguese; this file and the
README are in English. Keep new code consistent with the file it lives in.

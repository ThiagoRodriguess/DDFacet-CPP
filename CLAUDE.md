# CLAUDE.md

## Commands

```bash
# pipeline (no external deps)
g++ -std=c++17 -O2 -Wall -Wextra -fopenmp -Iinclude src/main_ompc.cpp \
    -o build/ddfacet_ompc

# MS exporter (needs casacore)
g++ -std=c++17 -O2 -Iinclude -isystem /usr/include/casacore \
    tools/ms_export.cpp src/ms_io.cpp \
    -lcasa_ms -lcasa_tables -lcasa_casa -o build/ms_export

# run
./build/ms_export <path.ms> data/obs
./build/ddfacet_ompc data/obs <npix> <major cycles> <nchan>

# invariance check — the three checksums must match exactly
for n in 1 2 4; do
    OMP_NUM_THREADS=$n ./build/ddfacet_ompc data/obs 128 2 4 | grep checksum
done

# w-term A/B
./scripts/demo_wterm.sh
```

Run the invariance check before trusting any timing number. Divergence beyond
float rounding (~1e-6 relative) means a lost or duplicated visibility, or a
write race.

## Gotchas

- **Never pipe compiler output through `head`** — it closes the pipe, `g++`
  dies of SIGPIPE, and the stale binary survives looking like a good build.
  Redirect to a file.
- **`npix` must be a power of two** (the FFT is radix-2).
- **Inside `omp target`: POD only.** No `std::vector`, `std::complex`,
  casacore, FFTW or `std::cout`. `printf` works. Functions called from inside
  must sit between `#pragma omp declare target` / `end declare target`.
- **Do not wrap the `target` region in an explicit `omp task`.** It makes the
  channels run in parallel under `g++`, but diverges from the form the OMPC
  runtime expects.
- **Under `g++` the channels run sequentially** — no offload device, and
  `nowait` permits but does not require deferred execution. 1, 2 and 4 threads
  give the same wall time. Not a bug; real distribution needs the OMPC runtime.
- **`read_ms` must use `CHAN_FREQ[channel]`, not `[0]`.** The wavelength
  converts UVW from metres to wavelengths; the wrong one silently corrupts
  `(u, v)`. Across `sim_large.ms` that is ~2% in `u_max`.
- **Keep the direction-dependent gain deterministic** — a function of
  `(l0, m0)` only. A random gain would differ per node and break invariance.
- **The w-term and the DDE are zero at the phase centre by construction**
  (`l0 = m0 = 0` → `n0 = 1`). Use `DDF_OFFSET` to make either visible; testing
  them at the centre proves nothing.

## Data

Measurement Sets are not versioned.

- `tools/make_ms.py` generates a synthetic MS (needs `python-casacore`).
  Its `w` is **exactly zero** — useless for the w-term.
- GLEAM / SKA1-Low: `sim_large.ms` has 4 channels and real `w`;
  `sim_small.ms` has one channel, so it cannot exercise the channel axis.

## Cluster

`scripts/sorgan_ompc.sh` handles setup, build and run. The launch is MPMD:
N `llvm-offload-mpi-proxy-device` processes plus one running the program,
separated by `:` in the `mpirun` line. An MPI implementation must be loaded
(`ml load mpich`) even though no MPI call appears in the source.

Offload is working when the `[WORKER]` lines print PIDs different from
`[HEAD]`. Identical PIDs mean it fell back to local execution.

## Conventions

Source comments and identifiers are in Portuguese; this file and the README are
in English. Match the file you are editing.

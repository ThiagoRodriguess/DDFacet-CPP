#!/bin/bash
# =============================================================================
#  DDFacet on OpenMP Cluster — build and distributed run
# =============================================================================
#
#  WHAT THIS SCRIPT DOES
#  ---------------------
#  Builds and runs src/main_ompc.cpp, which distributes the DDFacet kernel
#  (degrid + residual + grid) BY CHANNEL using `#pragma omp target`. Under
#  OMPC each `target` region is dispatched by the runtime to a cluster NODE
#  over MPI.
#
#  EXECUTION MODEL (needed to read the `run` command)
#  --------------------------------------------------
#  OMPC uses an MPMD launch with TWO process groups:
#
#     mpirun -np N  ...  llvm-offload-mpi-proxy-device   \  <- N "devices": the
#            :                                              nodes that execute
#            -np 1  ...  ./ddfacet_ompc                     the target regions
#                                                        ^- 1 "head": runs the
#                                                           program itself
#
#  So N+1 processes in total. The head creates the tasks; the proxy devices
#  receive and execute them. That is what the ':' in the command separates.
#
#  PREREQUISITES
#  -------------
#    * cluster access
#    * work under /scratch/$USER, not your home directory
#    * pull the container FROM A COMPUTE NODE (the login node has network and
#      quota restrictions)
#
#  USAGE
#  -----
#    ./sorgan_ompc.sh setup           # pull the OMPC container (once)
#    ./sorgan_ompc.sh build           # compile with the container's clang
#    ./sorgan_ompc.sh run  <hosts> [nchan] [nvis]
#    ./sorgan_ompc.sh check <hosts>   # INVARIANCE TEST (1 node vs N nodes)
#
#  Full session:
#    salloc -N 2 -p cpu --no-shell --exclusive --time 01:00:00
#    # note the allocated nodes, e.g. sorgan-cpu1,sorgan-cpu2
#    ssh sorgan-cpu1
#    cd /scratch/$USER/ddfacet_cpp
#    ./scripts/sorgan_ompc.sh setup
#    ./scripts/sorgan_ompc.sh build
#    ./scripts/sorgan_ompc.sh check sorgan-cpu1,sorgan-cpu2
# =============================================================================

set -u

# --- Configuration (adjust if the names differ) ------------------------------
CONTAINER_URL="docker://cl3t0/ompc:latest"
SIF="${SIF:-ompc_latest.sif}"          # file produced by the pull
SRC="src/main_ompc.cpp"
BIN="ddfacet_ompc"
INC="include"
MPI_MODULE="${MPI_MODULE:-mpich}"
# --nv targets NVIDIA GPUs; harmless on CPU nodes. Clear it if it causes trouble.
APPTAINER_FLAGS="${APPTAINER_FLAGS:---nv}"

msg()  { printf '\n\033[1;36m>>> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warning]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# This script uses RELATIVE paths (src/, include/), so it must run from the
# project ROOT, not from inside scripts/.
require_root() {
    if [ ! -f "$SRC" ] || [ ! -d "$INC" ]; then
        die "run from the project ROOT (where src/ and include/ live).
       You are in: $(pwd)
       Try:  cd \"\$(dirname \"\$0\")/..\"  &&  ./scripts/$(basename "$0") ..."
    fi
}

require_sif() {
    [ -f "$SIF" ] || die "container '$SIF' not found. Run: $0 setup
       (check the name of the .sif produced — it may differ; export SIF=<name>.sif)"
}

# --- setup: pull the container ----------------------------------------------
cmd_setup() {
    msg "Pulling the OMPC container ($CONTAINER_URL)"
    warn "Do this from a COMPUTE NODE, not the login node:"
    warn "  salloc -N 1 -p cpu --no-shell --exclusive --time 01:00:00"
    warn "  ssh <allocated-node>"

    command -v apptainer >/dev/null 2>&1 || die "apptainer not found (try: ml load apptainer)"
    apptainer pull "$CONTAINER_URL" || die "container pull failed"

    msg "Available .sif files:"
    ls -la ./*.sif 2>/dev/null || warn "no .sif in the current directory"
    warn "If the file has a different name: export SIF=<name>.sif"
}

# --- build: compile with the container's patched clang ----------------------
cmd_build() {
    require_root; require_sif

    msg "Compiling $SRC with the container's clang++"
    # -fopenmp-targets=x86_64-pc-linux-gnu: the "device" is another x86 NODE,
    # not a GPU. That flag is what enables dispatch over the network.
    apptainer exec $APPTAINER_FLAGS "$SIF" \
        clang++ -O2 -std=c++17 -fopenmp \
                -fopenmp-targets=x86_64-pc-linux-gnu \
                -I"$INC" "$SRC" -o "$BIN" \
        || die "compilation failed"

    msg "OK: '$BIN' built"
    ls -la "$BIN"
}

# --- run: distributed execution (N proxy devices + 1 head) ------------------
# usage: cmd_run <comma-separated-hosts> [npix] [cycles] [nchan]
cmd_run() {
    require_root; require_sif
    local hosts="${1:-}"
    local npix="${2:-128}"
    local cycles="${3:-2}"
    local nchan="${4:-4}"
    [ -n "$hosts" ] || die "give the hosts. e.g.: $0 run sorgan-cpu1,sorgan-cpu2"
    [ -x "$BIN" ]   || die "'$BIN' not built. Run: $0 build"

    # number of devices = number of hosts given
    local ndev
    ndev=$(awk -F',' '{print NF}' <<< "$hosts")

    msg "Running on $ndev offload node(s)  |  npix=$npix cycles=$cycles channels=$nchan"
    echo "    hosts: $hosts"

    module load "$MPI_MODULE" 2>/dev/null || warn "could not 'module load $MPI_MODULE' (fine if already in the environment)"

    # MPMD: N proxy devices  :  1 head running our program
    mpirun -hosts="$hosts" \
           -np "$ndev" apptainer exec --sharens $APPTAINER_FLAGS "$SIF" llvm-offload-mpi-proxy-device \
         : -np 1        apptainer exec --sharens $APPTAINER_FLAGS "$SIF" ./"$BIN" data/obs "$npix" "$cycles" "$nchan"
}

# --- check: INVARIANCE TEST — the result must not depend on the node count --
cmd_check() {
    local hosts="${1:-}"
    [ -n "$hosts" ] || die "give the hosts. e.g.: $0 check sorgan-cpu1,sorgan-cpu2"

    local first
    first=$(cut -d',' -f1 <<< "$hosts")

    msg "INVARIANCE TEST — the checksums must be IDENTICAL"
    echo "  Distributing only splits the channels; summing the partial UV grids"
    echo "  must give the same result. If the checksums differ, the decomposition"
    echo "  is wrong (a channel lost, duplicated, or a write race)."

    msg "[1/2] ONE offload node ($first)"
    cmd_run "$first" 128 2 4 | tee /tmp/ompc_1node.log

    msg "[2/2] ALL nodes ($hosts)"
    cmd_run "$hosts" 128 2 4 | tee /tmp/ompc_Nnodes.log

    msg "Checksum comparison"
    local c1 cN
    c1=$(grep -i 'checksum' /tmp/ompc_1node.log  | tail -1)
    cN=$(grep -i 'checksum' /tmp/ompc_Nnodes.log | tail -1)
    echo "  1 node : $c1"
    echo "  N nodes: $cN"
    if [ "$c1" = "$cN" ] && [ -n "$c1" ]; then
        printf '\n\033[1;32m  OK — the distribution is correct.\033[0m\n\n'
    else
        printf '\n\033[1;31m  MISMATCH — investigate the decomposition/mapping.\033[0m\n\n'
        return 1
    fi
}

# --- dispatch ---------------------------------------------------------------
case "${1:-}" in
    setup) shift; cmd_setup "$@" ;;
    build) shift; cmd_build "$@" ;;
    run)   shift; cmd_run   "$@" ;;
    check) shift; cmd_check "$@" ;;
    *)
        sed -n '2,48p' "$0"
        echo
        echo "Commands: setup | build | run <hosts> [npix] [cycles] [nchan] | check <hosts>"
        ;;
esac

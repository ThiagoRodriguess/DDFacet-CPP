/**
 * @file mpi_util.h
 * @brief Camada multi-nó (MPI) ISOLADA atrás de #ifdef USE_MPI.
 *
 * Eixo de paralelização: os Measurement Sets (J). Cada rank MPI processa um
 * subconjunto dos MS, faz predict/grid nas suas grades UV locais, e ao final do
 * passo Grid um MPI_Allreduce SOMA as grades UV entre todos os ranks — que é
 * exatamente o "g = Grid(δv)" do Algoritmo 1 acumulado sobre todos os MS.
 *
 * Sem -DUSE_MPI: tudo vira no-op (rank 0, size 1, all-reduce = identidade), e o
 * binário processa todos os MS num único processo (caminho sequencial testável).
 * Com -DUSE_MPI (mpic++): o laço de MS é distribuído entre os ranks.
 *
 * Modelo híbrido: MPI entre nós (eixo J) + OpenMP dentro do nó (facetas/vis).
 * Como as chamadas MPI ficam FORA das regiões OpenMP, basta MPI_THREAD_FUNNELED.
 */
#ifndef MPI_UTIL_H
#define MPI_UTIL_H

#include <cstddef>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace ddfacet {

inline void mpi_init(int& argc, char**& argv) {
#ifdef USE_MPI
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
#else
    (void)argc; (void)argv;
#endif
}

inline void mpi_finalize() {
#ifdef USE_MPI
    MPI_Finalize();
#endif
}

inline int mpi_rank() {
#ifdef USE_MPI
    int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r;
#else
    return 0;
#endif
}

inline int mpi_size() {
#ifdef USE_MPI
    int s = 1; MPI_Comm_size(MPI_COMM_WORLD, &s); return s;
#else
    return 1;
#endif
}

inline void mpi_barrier() {
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

/** @brief Soma in-place um buffer de `n` floats entre todos os ranks (all-reduce). */
inline void mpi_allreduce_floats(float* data, std::size_t n) {
#ifdef USE_MPI
    MPI_Allreduce(MPI_IN_PLACE, data, static_cast<int>(n), MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#else
    (void)data; (void)n;
#endif
}

/** @brief Soma um escalar double entre todos os ranks e devolve o total. */
inline double mpi_allreduce_double(double v) {
#ifdef USE_MPI
    double out = 0.0; MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); return out;
#else
    return v;
#endif
}

} // namespace ddfacet

#endif // MPI_UTIL_H

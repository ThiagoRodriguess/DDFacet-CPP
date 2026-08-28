/**
 * @file ompc_kernel.h
 * @brief Kernel de degrid+residual+grid POD-only, offloadável via OpenMP target.
 *
 * ─── Por que este arquivo existe ──────────────────────────────────────────────
 * No OpenMP Cluster (OMPC) a unidade de distribuição é a REGIÃO `omp target`:
 * o runtime empacota a região e a envia para um nó remoto via MPI. Isso impõe
 * restrições que o resto do pipeline (C++ idiomático) não satisfaz:
 *
 *   1. Os dados que cruzam a fronteira precisam ser MAPEÁVEIS (`map(to:/from:)`),
 *      isto é, arrays POD contíguos — nada de std::vector, std::complex,
 *      std::string ou ponteiros para ponteiros.
 *   2. Nada de I/O nem de bibliotecas de host dentro da região (casacore, FFTW,
 *      std::cout): a leitura do MS e as FFTs ficam no HOST.
 *   3. Toda função chamada de dentro do target precisa estar em `declare target`.
 *
 * Por isso o kernel abaixo é escrito em C puro sobre arrays planos, com números
 * complexos representados como PARES DE ARRAYS (re/im) — a forma mais portável
 * de atravessar a fronteira de offload.
 *
 * ─── Onde ele entra no Algoritmo 1 ────────────────────────────────────────────
 * Por CANAL (a unidade de trabalho distribuível — "canal c → nó c"):
 *
 *      HOST                          TARGET (nó remoto)             HOST
 *   FFT(modelo) ──map(to)──►  degrid → residual → grid  ──map(from)──►  Σ canais
 *                                                                      → FFT⁻¹
 *                                                                      → CLEAN
 *
 * As três etapas são fundidas em UMA passada sobre as visibilidades: assim a
 * visibilidade predita nunca é materializada nem transferida de volta, o que
 * reduz o tráfego entre nós (que no OMPC é o custo dominante).
 */
#ifndef OMPC_KERNEL_H
#define OMPC_KERNEL_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <math.h>

#ifndef OMPC_PI
#define OMPC_PI 3.14159265358979323846
#endif

/*
 * `declare target` torna as funções abaixo compiláveis para o "device" (que no
 * OMPC é outro NÓ do cluster, não uma GPU).
 */
#pragma omp declare target

/** @brief Índice periódico na grade UV (equivalente ao wrap_index do host). */
static inline int ompc_wrap(int i, int n) {
    while (i < 0)   i += n;
    while (i >= n)  i -= n;
    return i;
}

/**
 * @brief degrid + residual + grid de UM canal, para UMA faceta.
 *
 * Funde as três etapas numa única varredura das visibilidades:
 *   v̂_k = DEGRID(modelo)·exp(-2πi(u·l₀+v·m₀+w·(n₀-1)))·G
 *   δv_k = v_k − v̂_k
 *   grade += δv_k·exp(+2πi(u·l₀+v·m₀+w·(n₀-1)))·conj(G)
 *
 * @param u,v,w         coordenadas [comprimentos de onda], nvis elementos
 * @param vis_re,vis_im visibilidades medidas v_k
 * @param flag          1 = descartar a visibilidade
 * @param nvis          número de visibilidades deste canal/fatia
 * @param mdl_re,mdl_im grade UV do modelo (FFT do modelo, feita no host) [nx*ny]
 * @param grid_re,grid_im  SAÍDA: grade UV do resíduo, ACUMULADA [nx*ny]
 * @param nx,ny         dimensões da grade da faceta
 * @param cell          tamanho do pixel [rad]
 * @param l0,m0,n0m1    centro de fase da faceta e o termo-w (n₀−1)
 * @param gain_re,gain_im  ganho dependente da direção G da faceta (DDE)
 * @param W             meio-suporte do kernel de gridding
 * @param sigma         largura do kernel gaussiano
 */
static inline void ompc_degrid_residual_grid(
        const double* u, const double* v, const double* w,
        const float* vis_re, const float* vis_im,
        const unsigned char* flag, int nvis,
        const float* mdl_re, const float* mdl_im,
        float* grid_re, float* grid_im,
        int nx, int ny, double cell,
        double l0, double m0, double n0m1,
        double gain_re, double gain_im,
        int W, double sigma)
{
    const double s2 = 2.0 * sigma * sigma;

    for (int k = 0; k < nvis; ++k) {
        if (flag[k]) continue;

        /* (u,v) [λ] → posição contínua em pixels na grade */
        const double ix_c = u[k] * nx * cell + nx / 2.0;
        const double iy_c = v[k] * ny * cell + ny / 2.0;
        const int    ix0  = (int)lround(ix_c);
        const int    iy0  = (int)lround(iy_c);

        /* ── DEGRID: interpola a grade do modelo na posição da visibilidade ── */
        double acc_re = 0.0, acc_im = 0.0, wsum = 0.0;
        for (int dj = -W; dj <= W; ++dj) {
            const int    iy = iy0 + dj;
            const double dv = (double)iy - iy_c;
            const int    gy = ompc_wrap(iy, ny);
            for (int di = -W; di <= W; ++di) {
                const int    ix = ix0 + di;
                const double du = (double)ix - ix_c;
                const double kw = exp(-(du * du + dv * dv) / s2);
                const int    gx = ompc_wrap(ix, nx);
                const int    id = gy * nx + gx;
                acc_re += mdl_re[id] * kw;
                acc_im += mdl_im[id] * kw;
                wsum   += kw;
            }
        }
        if (wsum > 0.0) { acc_re /= wsum; acc_im /= wsum; }

        /* Fase da faceta COM termo-w: exp(-2πi(u·l₀ + v·m₀ + w·(n₀−1))) */
        const double ph = -2.0 * OMPC_PI * (u[k] * l0 + v[k] * m0 + w[k] * n0m1);
        const double cp = cos(ph), sp = sin(ph);
        double p_re = acc_re * cp - acc_im * sp;
        double p_im = acc_re * sp + acc_im * cp;

        /* DDE: × G */
        const double g_re = p_re * gain_re - p_im * gain_im;
        const double g_im = p_re * gain_im + p_im * gain_re;
        p_re = g_re; p_im = g_im;

        /* ── RESIDUAL: δv = v − v̂ ──────────────────────────────────────────── */
        double d_re = (double)vis_re[k] - p_re;
        double d_im = (double)vis_im[k] - p_im;

        /* ── GRID (adjunto): fase conjugada e ×conj(G) ─────────────────────── */
        const double cq = cp, sq = -sp;               /* exp(+2πi(...)) */
        double q_re = d_re * cq - d_im * sq;
        double q_im = d_re * sq + d_im * cq;
        /* conj(G) = (gain_re, -gain_im) */
        d_re = q_re * gain_re + q_im * gain_im;
        d_im = q_im * gain_re - q_re * gain_im;

        for (int dj = -W; dj <= W; ++dj) {
            const int    iy = iy0 + dj;
            const double dv = (double)iy - iy_c;
            const int    gy = ompc_wrap(iy, ny);
            for (int di = -W; di <= W; ++di) {
                const int    ix = ix0 + di;
                const double du = (double)ix - ix_c;
                const double kw = exp(-(du * du + dv * dv) / s2);
                const int    gx = ompc_wrap(ix, nx);
                const int    id = gy * nx + gx;
                grid_re[id] += (float)(d_re * kw);
                grid_im[id] += (float)(d_im * kw);
            }
        }
    }
}

#pragma omp end declare target

#endif /* OMPC_KERNEL_H */

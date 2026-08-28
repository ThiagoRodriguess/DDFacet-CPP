/**
 * @file ompc_kernel.h
 * @brief POD-only degrid + residual + grid kernel, offloadable via omp target.
 *
 * --- Why this file exists ---------------------------------------------------
 * Under OpenMP Cluster the unit of distribution is the `omp target` region: the
 * runtime packs it up and ships it to a remote node over MPI. That imposes
 * constraints the rest of a C++ codebase does not satisfy:
 *
 *   1. Data crossing the boundary must be MAPPABLE (`map(to:/from:)`), i.e.
 *      contiguous POD arrays — no std::vector, std::complex, std::string or
 *      pointer-to-pointer.
 *   2. No I/O and no host libraries inside the region (casacore, FFTW,
 *      std::cout): MS reading and the FFTs stay on the HOST.
 *   3. Every function called from inside must be inside `declare target`.
 *
 * So the kernel below is written in plain C over flat arrays, with complex
 * numbers represented as PAIRS OF ARRAYS (re/im) — the most portable way to
 * cross an offload boundary.
 *
 * --- Where it sits in the algorithm -----------------------------------------
 * Per channel (the distributable unit of work — "channel c -> node c"):
 *
 *      HOST                          TARGET (remote node)              HOST
 *   FFT(model) ---map(to)--->  degrid -> residual -> grid  ---map(from)---> sum
 *                                                                        -> FFT-1
 *                                                                        -> CLEAN
 *
 * The three steps are fused into ONE pass over the visibilities: the predicted
 * visibility is never materialised nor sent back, which cuts the inter-node
 * traffic that dominates the cost under OMPC.
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
 * `declare target` makes the functions below compilable for the "device",
 * which under OMPC is another CLUSTER NODE, not a GPU.
 */
#pragma omp declare target

/** @brief Periodic index into the UV grid. */
static inline int ompc_wrap(int i, int n) {
    while (i < 0)   i += n;
    while (i >= n)  i -= n;
    return i;
}

/**
 * @brief degrid + residual + grid for ONE channel and ONE facet.
 *
 * Fuses the three steps into a single sweep over the visibilities:
 *   v_pred = DEGRID(model) * exp(-2i*pi*(u*l0 + v*m0 + w*(n0-1))) * G
 *   dv     = v_meas - v_pred
 *   grid  += dv * exp(+2i*pi*(u*l0 + v*m0 + w*(n0-1))) * conj(G)
 *
 * @param u,v,w         coordinates [wavelengths], nvis elements each
 * @param vis_re,vis_im measured visibilities
 * @param flag          1 = skip this visibility
 * @param nvis          number of visibilities in this channel/slice
 * @param mdl_re,mdl_im model UV grid (FFT of the model, computed on the host)
 * @param grid_re,grid_im  OUTPUT: residual UV grid, ACCUMULATED [nx*ny]
 * @param nx,ny         facet grid dimensions
 * @param cell          pixel size [rad]
 * @param l0,m0,n0m1    facet phase centre and the w-term (n0 - 1)
 * @param gain_re,gain_im  direction-dependent gain G of this facet
 * @param W             gridding kernel half-support
 * @param sigma         Gaussian kernel width
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

        /* (u,v) [wavelengths] -> continuous pixel position on the grid */
        const double ix_c = u[k] * nx * cell + nx / 2.0;
        const double iy_c = v[k] * ny * cell + ny / 2.0;
        const int    ix0  = (int)lround(ix_c);
        const int    iy0  = (int)lround(iy_c);

        /* -- DEGRID: interpolate the model grid at the visibility position -- */
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

        /* Facet phase shift, including the w-term:
           exp(-2i*pi*(u*l0 + v*m0 + w*(n0-1))) */
        const double ph = -2.0 * OMPC_PI * (u[k] * l0 + v[k] * m0 + w[k] * n0m1);
        const double cp = cos(ph), sp = sin(ph);
        double p_re = acc_re * cp - acc_im * sp;
        double p_im = acc_re * sp + acc_im * cp;

        /* Direction-dependent gain: multiply by G */
        const double g_re = p_re * gain_re - p_im * gain_im;
        const double g_im = p_re * gain_im + p_im * gain_re;
        p_re = g_re; p_im = g_im;

        /* -- RESIDUAL: dv = v_meas - v_pred ------------------------------- */
        double d_re = (double)vis_re[k] - p_re;
        double d_im = (double)vis_im[k] - p_im;

        /* -- GRID (adjoint): conjugate phase and multiply by conj(G) ------ */
        const double cq = cp, sq = -sp;               /* exp(+2i*pi*(...)) */
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

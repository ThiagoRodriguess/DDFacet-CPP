/**
 * @file main_ompc.cpp
 * @brief DDFacet major-cycle loop, distributed by channel with OpenMP Cluster.
 *
 * Each spectral channel is an independent unit of work with its own frequency,
 * so each becomes one `omp target` region that the OMPC runtime dispatches to
 * a cluster node:
 *
 *   HOST                    TARGET (one per channel, on distinct nodes)   HOST
 *   FFT(model) --map(to)--> degrid -> residual -> grid --map(from)-->  sum
 *                                                                   -> FFT-1
 *                                                                   -> CLEAN
 *                                                                   -> repeat
 *
 * Reading, transforming and deconvolving all happen on the host; only the
 * per-visibility work is offloaded. That work lives in ompc_kernel.h, which
 * fuses degrid, residual and grid into a single sweep so the predicted
 * visibilities never travel back across the network.
 *
 * Usage:
 *   ./ddfacet_ompc <vis_prefix> [npix] [cycles] [nchan]
 *   e.g. ./ddfacet_ompc data/obs 128 3 4   reads data/obs_ch0.vis .. _ch3.vis
 *
 * Output: dirty_ompc.fits and model_ompc.fits.
 */
#include "fft.h"
#include "fits.h"
#include "ompc_kernel.h"
#include "vis_file.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

using namespace ddfacet;

namespace {

// Gridding kernel: Gaussian, half-support W, width sigma.
constexpr int    KERNEL_W  = 5;
constexpr double KERNEL_SG = 1.0;
constexpr double ARCSEC    = 4.8481368110953599e-06;

// Deconvolution.
constexpr int    N_MINOR    = 200;
constexpr double CLEAN_GAIN = 0.1;

/** @brief Read an environment variable as a double, or return `fallback`. */
double env_double(const char* name, double fallback) {
    const char* e = std::getenv(name);
    return e ? std::atof(e) : fallback;
}

/** @brief Largest value in an image. */
float peak_of(const std::vector<float>& img) {
    float p = 0.0f;
    for (float v : img) if (v > p) p = v;
    return p;
}

/** @brief Centre the origin, transform, and centre it again. */
void transform_centred(std::vector<float>& re, std::vector<float>& im,
                       int nx, int ny, FFTDirection dir) {
    fftshift(re, nx, ny); fftshift(im, nx, ny);
    fft2d(re, im, nx, ny, dir);
    fftshift(re, nx, ny); fftshift(im, nx, ny);
}

/**
 * @brief Hogbom CLEAN: repeatedly subtract the beam scaled to the current peak.
 *
 * Consumes `dirty` in place and accumulates the components into `model`.
 */
void hogbom_clean(std::vector<float>& dirty, std::vector<float>& model,
                  const std::vector<float>& psf, float beam_peak,
                  int nx, int ny) {
    for (int it = 0; it < N_MINOR; ++it) {
        float peak = 0.0f;
        int px = 0, py = 0;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const float a = dirty[static_cast<std::size_t>(j) * nx + i];
                if (std::fabs(a) > std::fabs(peak)) { peak = a; px = i; py = j; }
            }
        if (std::fabs(peak) < 1e-9f) break;

        const float delta = static_cast<float>(CLEAN_GAIN * peak);
        model[static_cast<std::size_t>(py) * nx + px] += delta;

        // Subtract the PSF centred on the peak.
        const int cx = nx / 2, cy = ny / 2;
        for (int j = 0; j < ny; ++j) {
            const int pj = j - py + cy;
            if (pj < 0 || pj >= ny) continue;
            for (int i = 0; i < nx; ++i) {
                const int pi = i - px + cx;
                if (pi < 0 || pi >= nx) continue;
                dirty[static_cast<std::size_t>(j) * nx + i] -=
                    delta * psf[static_cast<std::size_t>(pj) * nx + pi] / beam_peak;
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <vis_prefix> [npix] [cycles] [nchan]\n"
            "  e.g. %s data/obs 128 3 4\n", argv[0], argv[0]);
        return 1;
    }
    const std::string prefix  = argv[1];
    const int         npix    = (argc > 2) ? std::atoi(argv[2]) : 256;
    const int         ncycles = (argc > 3) ? std::atoi(argv[3]) : 5;
    const int         nchan   = (argc > 4) ? std::atoi(argv[4]) : 4;

    if (!is_power_of_two(npix)) {
        std::fprintf(stderr, "[error] npix must be a power of two (got %d)\n", npix);
        return 1;
    }

    std::printf("========================================================\n");
    std::printf("  DDFacet OMPC - major cycles distributed BY CHANNEL\n");
    std::printf("  [HEAD] pid=%d\n", getpid());
    std::printf("========================================================\n");

    // -- Load one .vis per channel -------------------------------------------
    std::vector<VisFile> channels(nchan);
    double umax_global = 0.0;
    for (int c = 0; c < nchan; ++c) {
        char fn[1024];
        std::snprintf(fn, sizeof(fn), "%s_ch%d.vis", prefix.c_str(), c);
        if (!vis_file_read(fn, channels[c])) return 1;
        umax_global = std::max(umax_global, channels[c].umax_wl);
        std::printf("  channel %d: %8d vis | %.3f MHz | u_max %.1f lambda\n",
                    c, channels[c].nvis, channels[c].freq_hz / 1e6,
                    channels[c].umax_wl);
    }

    // Cell size from the largest u_max: Nyquist with a 3x margin.
    const double      cell = 1.0 / (3.0 * umax_global);
    const int         nx = npix, ny = npix;
    const std::size_t ng = static_cast<std::size_t>(nx) * ny;
    std::printf("  image  : %dx%d  cell=%.4f arcsec  cycles=%d\n",
                nx, ny, cell / ARCSEC, ncycles);

    // -- Phase centre ---------------------------------------------------------
    // At the origin n0 = 1, so the w-term w*(n0-1) vanishes by construction:
    // the term lives in the phase-centre shift, and a centred facet has none.
    // DDF_OFFSET moves the centre, which is what happens to every off-centre
    // facet in a faceted run. DDF_NOW zeroes the term for a side-by-side test.
    const double off_pix = env_double("DDF_OFFSET", 0.0);
    const double l0  = off_pix * cell;
    const double m0  = off_pix * cell;
    const double lm2 = l0 * l0 + m0 * m0;
    const bool   use_w = (std::getenv("DDF_NOW") == nullptr);
    const double n0m1  = (use_w && lm2 < 1.0) ? (std::sqrt(1.0 - lm2) - 1.0) : 0.0;

    // -- Direction-dependent gain (scalar RIME/Jones) -------------------------
    // Applied as G in the degrid and conj(G) in the adjoint, which preserves
    // the adjoint relation. It must be a deterministic function of (l0, m0):
    // a random gain would differ per node and break the invariance criterion.
    double     gain_re = 1.0, gain_im = 0.0;
    const bool use_dde = (std::getenv("DDF_DDE") != nullptr);
    if (use_dde) {
        const double r   = std::sqrt(lm2);
        const double amp = 1.0 / (1.0 + 40.0 * r * r);   // primary-beam analogue
        const double pha = 150.0 * r;                     // phase-error analogue
        gain_re = amp * std::cos(pha);
        gain_im = amp * std::sin(pha);
    }

    std::printf("  phase  : offset=%.0f px  l0=%.3e m0=%.3e  w-term n0-1=%.6e %s\n",
                off_pix, l0, m0, n0m1, use_w ? "" : "[DDF_NOW: w-term OFF]");
    std::printf("  DDE    : |G|=%.6f  arg(G)=%.6f rad %s\n",
                std::sqrt(gain_re * gain_re + gain_im * gain_im),
                std::atan2(gain_im, gain_re),
                use_dde ? "" : "[identity - set DDF_DDE=1]");

    // -- PSF: grid unit visibilities through the same kernel ------------------
    std::vector<float> psf_re(ng, 0.0f), psf_im(ng, 0.0f);
    {
        const std::vector<float> zero_model(ng, 0.0f);
        for (int c = 0; c < nchan; ++c) {
            const VisFile& k = channels[c];
            // A zero model makes dv = v; feeding v = 1 yields grid(1) = PSF.
            const std::vector<float> ones(k.nvis, 1.0f);
            const std::vector<float> zeros(k.nvis, 0.0f);
            ompc_degrid_residual_grid(k.u.data(), k.v.data(), k.w.data(),
                                      ones.data(), zeros.data(), k.flag.data(), k.nvis,
                                      zero_model.data(), zero_model.data(),
                                      psf_re.data(), psf_im.data(),
                                      nx, ny, cell, l0, m0, n0m1,
                                      gain_re, gain_im, KERNEL_W, KERNEL_SG);
        }
        transform_centred(psf_re, psf_im, nx, ny, FFT_INVERSE);
    }
    float beam_peak = peak_of(psf_re);
    if (beam_peak <= 0.0f) beam_peak = 1.0f;
    std::printf("  PSF    : peak S = %.6g\n", static_cast<double>(beam_peak));

    // -- Major cycles ---------------------------------------------------------
    std::vector<float> model(ng, 0.0f);
    std::vector<float> dirty_first;
    std::vector<std::vector<float>> grid_re(nchan), grid_im(nchan);
    for (int c = 0; c < nchan; ++c) {
        grid_re[c].assign(ng, 0.0f);
        grid_im[c].assign(ng, 0.0f);
    }

    const double t_start = omp_get_wtime();
    double       t_offload = 0.0;

    for (int cycle = 0; cycle < ncycles; ++cycle) {
        std::printf("\n=== MAJOR CYCLE %d ===\n", cycle);

        // (a) HOST: transform the model into a UV grid.
        std::vector<float> mdl_re(model), mdl_im(ng, 0.0f);
        transform_centred(mdl_re, mdl_im, nx, ny, FFT_FORWARD);

        for (int c = 0; c < nchan; ++c) {
            std::fill(grid_re[c].begin(), grid_re[c].end(), 0.0f);
            std::fill(grid_im[c].begin(), grid_im[c].end(), 0.0f);
        }

        // (b) OFFLOAD: one target region per channel.
        const double t0 = omp_get_wtime();
        #pragma omp parallel
        #pragma omp single
        {
            for (int c = 0; c < nchan; ++c) {
                const int            nv  = channels[c].nvis;
                const double*        up  = channels[c].u.data();
                const double*        vp  = channels[c].v.data();
                const double*        wp  = channels[c].w.data();
                const float*         vrp = channels[c].re.data();
                const float*         vip = channels[c].im.data();
                const unsigned char* fp  = channels[c].flag.data();
                const float*         mrp = mdl_re.data();
                const float*         mip = mdl_im.data();
                float*               grp = grid_re[c].data();
                float*               gip = grid_im[c].data();
                const int            ngi = static_cast<int>(ng);

                // Canonical OMPC form: `target nowait` plus `depend`. That is
                // how the runtime recognises the regions as independent tasks
                // and dispatches them to distinct nodes.
                //
                // Under g++ there is no offload device, so the region runs
                // inline and the channels end up sequential: `nowait` permits
                // deferred execution but does not require it.
                #pragma omp target nowait                                    \
                    depend(out: grp[0:ngi])                                  \
                    map(to: up[0:nv], vp[0:nv], wp[0:nv],                    \
                            vrp[0:nv], vip[0:nv], fp[0:nv],                  \
                            mrp[0:ngi], mip[0:ngi])                          \
                    map(tofrom: grp[0:ngi], gip[0:ngi])
                {
                    printf("[WORKER] cycle %d channel %d  pid=%d\n", cycle, c, getpid());
                    ompc_degrid_residual_grid(up, vp, wp, vrp, vip, fp, nv,
                                              mrp, mip, grp, gip,
                                              nx, ny, cell, l0, m0, n0m1,
                                              gain_re, gain_im, KERNEL_W, KERNEL_SG);
                }
            }
            #pragma omp taskwait
        }
        t_offload += omp_get_wtime() - t0;

        // (c) HOST: sum the per-channel grids.
        std::vector<float> sum_re(ng, 0.0f), sum_im(ng, 0.0f);
        for (int c = 0; c < nchan; ++c)
            for (std::size_t i = 0; i < ng; ++i) {
                sum_re[i] += grid_re[c][i];
                sum_im[i] += grid_im[c][i];
            }

        // (d) HOST: back to the image domain.
        transform_centred(sum_re, sum_im, nx, ny, FFT_INVERSE);

        std::vector<float> dirty(ng);
        for (std::size_t i = 0; i < ng; ++i) dirty[i] = sum_re[i] / beam_peak;
        if (cycle == 0) dirty_first = dirty;

        double dmin = dirty[0], dmax = dirty[0], dsum = 0.0;
        for (std::size_t i = 0; i < ng; ++i) {
            if (dirty[i] < dmin) dmin = dirty[i];
            if (dirty[i] > dmax) dmax = dirty[i];
            dsum += dirty[i];
        }
        std::printf("  dirty  : min=%.6g max=%.6g sum=%.6g\n", dmin, dmax, dsum);

        // (e) HOST: deconvolve.
        hogbom_clean(dirty, model, psf_re, beam_peak, nx, ny);

        double msum = 0.0;
        for (std::size_t i = 0; i < ng; ++i) msum += model[i];
        std::printf("  model  : sum=%.6g peak=%.6g\n", msum,
                    static_cast<double>(peak_of(model)));
    }

    const double t_total = omp_get_wtime() - t_start;

    // -- Output ---------------------------------------------------------------
    write_fits("dirty_ompc.fits", dirty_first, nx, ny);
    write_fits("model_ompc.fits", model, nx, ny);

    // The checksum must not depend on how many nodes did the work.
    double checksum = 0.0;
    for (std::size_t i = 0; i < ng; ++i) checksum += std::fabs(dirty_first[i]);

    std::printf("\n--------------------------------------------------------\n");
    std::printf("  total time           : %.3f s\n", t_total);
    std::printf("  time in tasks        : %.3f s (%.1f%%)\n",
                t_offload, 100.0 * t_offload / (t_total > 0 ? t_total : 1));
    std::printf("  checksum |dirty|     : %.6f\n", checksum);
    std::printf("  FITS: dirty_ompc.fits, model_ompc.fits\n");
    std::printf("--------------------------------------------------------\n");
    std::printf("  INVARIANT: the checksum must be IDENTICAL with 1 node and\n");
    std::printf("  with N nodes. That is the proof the distribution is correct.\n");
    return 0;
}

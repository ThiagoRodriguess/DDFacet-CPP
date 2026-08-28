/**
 * @file main_ompc.cpp
 * @brief DDFacet (Algorithm 1) on OpenMP Cluster - full major-cycle loop.
 *
 * --- What this program does -------------------------------------------------
 * Runs the DDFacet major cycles, distributing the work BY CHANNEL across the
 * cluster nodes via `#pragma omp target`. Each channel is an independent unit
 * of work with its own frequency - the "num_channel = 1 per archi. node" layout.
 *
 *   HOST                    TARGET (one per channel, on distinct nodes)  HOST
 *   ----                    ----------------------------------------     ----
 *   FFT(model) ---map(to)--> degrid -> residual -> grid ---map(from)---> sum
 *                                                                    -> FFT-1
 *                                                                    -> CLEAN
 *                                                                    repeat
 *
 * --- Two container constraints that shaped this design ----------------------
 * 1. There is NO casacore. MS reading happens outside, in `tools/ms_export`,
 *    which writes one .vis file per channel (plain POD). Here we only fread.
 * 2. FFTW cannot be assumed either, so the FFT is in-tree (iterative radix-2)
 *    with no external dependency. It requires power-of-two dimensions.
 *
 * --- Usage ------------------------------------------------------------------
 *    ./ddfacet_ompc <vis_prefix> [npix] [cycles] [nchan]
 *    e.g.: ./ddfacet_ompc data/large 256 5 4
 *         reads data/large_ch0.vis .. _ch3.vis, 256x256 image, 5 major cycles
 *
 * Output: dirty_ompc.fits (dirty image) and model_ompc.fits (CLEAN model).
 */
#include "ompc_kernel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <unistd.h>

/* --- Gridding kernel parameters ------------------------------------------ */
static const int    KERNEL_W  = 5;
static const double KERNEL_SG = 1.0;
static const double ARCSEC    = 4.8481368110953599e-06;

/* ======================= 1. Reading the .vis files ======================== */

struct ChannelVis {
    int    channel = 0;
    double freq_hz = 0.0;
    double umax_wl = 0.0;
    int    nvis    = 0;
    std::vector<double>        u, v, w;
    std::vector<float>         re, im;
    std::vector<unsigned char> flag;
};

/** @brief Load a .vis file written by tools/ms_export. */
static bool load_vis(const std::string& fn, ChannelVis& c) {
    std::FILE* f = std::fopen(fn.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "[error] could not open '%s'\n", fn.c_str()); return false; }

    char magic[8] = {0};
    std::int32_t nvis = 0, ch = 0;
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "DDFVIS01", 8) != 0) {
        std::fprintf(stderr, "[error] '%s' is not a DDFVIS01 file\n", fn.c_str());
        std::fclose(f); return false;
    }
    /* A truncated .vis would silently yield garbage - check every read. */
    std::size_t nread = 0;
    nread += std::fread(&nvis,      sizeof(std::int32_t), 1, f);
    nread += std::fread(&ch,        sizeof(std::int32_t), 1, f);
    nread += std::fread(&c.freq_hz, sizeof(double),       1, f);
    nread += std::fread(&c.umax_wl, sizeof(double),       1, f);
    if (nread != 4 || nvis <= 0) {
        std::fprintf(stderr, "[error] invalid header in '%s' (nvis=%d)\n", fn.c_str(), nvis);
        std::fclose(f); return false;
    }

    c.channel = ch;
    c.nvis    = nvis;
    c.u.resize(nvis); c.v.resize(nvis); c.w.resize(nvis);
    c.re.resize(nvis); c.im.resize(nvis); c.flag.resize(nvis);

    const std::size_t n = (std::size_t)nvis;
    std::size_t got = 0;
    got += std::fread(c.u.data(),    sizeof(double), n, f);
    got += std::fread(c.v.data(),    sizeof(double), n, f);
    got += std::fread(c.w.data(),    sizeof(double), n, f);
    got += std::fread(c.re.data(),   sizeof(float),  n, f);
    got += std::fread(c.im.data(),   sizeof(float),  n, f);
    got += std::fread(c.flag.data(), sizeof(unsigned char), n, f);

    const bool ok = (got == 6 * n) && (std::ferror(f) == 0);
    if (!ok)
        std::fprintf(stderr, "[error] '%s' truncated: read %zu of %zu elements\n",
                     fn.c_str(), got, 6 * n);
    std::fclose(f);
    return ok;
}

/* ===================== 2. In-tree FFT (no FFTW) ========================== */

/** @brief Iterative radix-2 1-D FFT, in place. sign=-1 forward, +1 inverse. */
static void fft1d(float* re, float* im, int n, int sign) {
    /* bit-reversal */
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = sign * 2.0 * OMPC_PI / len;
        const double wr = cos(ang), wi = sin(ang);
        for (int i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;
            for (int k = 0; k < len / 2; ++k) {
                const int a = i + k, b = i + k + len / 2;
                const double xr = re[b] * cr - im[b] * ci;
                const double xi = re[b] * ci + im[b] * cr;
                re[b] = (float)(re[a] - xr); im[b] = (float)(im[a] - xi);
                re[a] = (float)(re[a] + xr); im[a] = (float)(im[a] + xi);
                const double ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr; cr = ncr;
            }
        }
    }
    if (sign > 0) for (int i = 0; i < n; ++i) { re[i] /= n; im[i] /= n; }
}

/** @brief Separable 2-D FFT (rows then columns). n must be a power of two. */
static void fft2d(std::vector<float>& re, std::vector<float>& im, int nx, int ny, int sign) {
    std::vector<float> lr(nx), li(nx);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) { lr[i] = re[(size_t)j*nx+i]; li[i] = im[(size_t)j*nx+i]; }
        fft1d(lr.data(), li.data(), nx, sign);
        for (int i = 0; i < nx; ++i) { re[(size_t)j*nx+i] = lr[i]; im[(size_t)j*nx+i] = li[i]; }
    }
    std::vector<float> cr(ny), ci(ny);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) { cr[j] = re[(size_t)j*nx+i]; ci[j] = im[(size_t)j*nx+i]; }
        fft1d(cr.data(), ci.data(), ny, sign);
        for (int j = 0; j < ny; ++j) { re[(size_t)j*nx+i] = cr[j]; im[(size_t)j*nx+i] = ci[j]; }
    }
}

/** @brief Swap quadrants (fftshift): moves the origin to the centre. */
static void fftshift(std::vector<float>& a, int nx, int ny) {
    std::vector<float> t(a.size());
    const int hx = nx / 2, hy = ny / 2;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            t[(size_t)((j + hy) % ny) * nx + (i + hx) % nx] = a[(size_t)j * nx + i];
    a.swap(t);
}

/* ==================== 3. FITS output (no dependencies) =================== */

static bool write_fits(const char* fn, const std::vector<float>& img, int nx, int ny) {
    std::FILE* f = std::fopen(fn, "wb");
    if (!f) return false;
    int ncards = 0;
    auto card = [&](const char* s) {
        char buf[81]; std::memset(buf, ' ', 80);
        std::size_t L = std::strlen(s); if (L > 80) L = 80;
        std::memcpy(buf, s, L); std::fwrite(buf, 1, 80, f); ++ncards;
    };
    char tmp[81];
    card("SIMPLE  =                    T");
    card("BITPIX  =                  -32");
    card("NAXIS   =                    2");
    std::snprintf(tmp, sizeof(tmp), "NAXIS1  = %20d", nx); card(tmp);
    std::snprintf(tmp, sizeof(tmp), "NAXIS2  = %20d", ny); card(tmp);
    card("BSCALE  =                  1.0");
    card("BZERO   =                  0.0");
    card("END");
    while (ncards % 36 != 0) card("");

    std::size_t nb = 0;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            float val = img[(size_t)j * nx + i];
            std::uint32_t u; std::memcpy(&u, &val, 4);
            unsigned char be[4] = { (unsigned char)((u>>24)&0xFF), (unsigned char)((u>>16)&0xFF),
                                    (unsigned char)((u>>8)&0xFF), (unsigned char)(u&0xFF) };
            std::fwrite(be, 1, 4, f); nb += 4;
        }
    while (nb % 2880) { char z = 0; std::fwrite(&z, 1, 1, f); ++nb; }
    std::fclose(f);
    return true;
}

/* ================================ 4. main ================================ */

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <vis_prefix> [npix] [cycles] [nchan]\n"
            "  e.g. %s data/large 256 5 4\n", argv[0], argv[0]);
        return 1;
    }
    const std::string prefix = argv[1];
    int npix   = (argc > 2) ? std::atoi(argv[2]) : 256;
    int ncycles = (argc > 3) ? std::atoi(argv[3]) : 5;
    int nchan  = (argc > 4) ? std::atoi(argv[4]) : 4;

    /* a FFT radix-2 exige potência de 2 */
    if (npix & (npix - 1)) {
        std::fprintf(stderr, "[error] npix must be a power of two (got %d)\n", npix);
        return 1;
    }

    std::printf("========================================================\n");
    std::printf("  DDFacet OMPC - full cycle, distributed BY CHANNEL\n");
    std::printf("  [HEAD] pid=%d\n", getpid());
    std::printf("========================================================\n");

    /* -- Load one .vis per channel ----------------------------------------- */
    std::vector<ChannelVis> channels(nchan);
    double umax_global = 0.0;
    for (int c = 0; c < nchan; ++c) {
        char fn[1024];
        std::snprintf(fn, sizeof(fn), "%s_ch%d.vis", prefix.c_str(), c);
        if (!load_vis(fn, channels[c])) return 1;
        if (channels[c].umax_wl > umax_global) umax_global = channels[c].umax_wl;
        std::printf("  channel %d: %8d vis | %.3f MHz | u_max %.1f lambda\n",
                    c, channels[c].nvis, channels[c].freq_hz / 1e6, channels[c].umax_wl);
    }

    /* cell size from the largest u_max (Nyquist with a 3x margin) */
    const double cell = 1.0 / (3.0 * umax_global);
    const int    nx = npix, ny = npix;
    const size_t ng = (size_t)nx * ny;
    std::printf("  image  : %dx%d  cell=%.4f arcsec  cycles=%d\n",
                nx, ny, cell / ARCSEC, ncycles);

    /* -- Facet phase centre --------------------------------------------------
     * By default the facet sits at the origin (l0=m0=0), so n0 = 1 and the
     * W-TERM vanishes - by construction, not by omission: the w*(n0-1) term
     * lives in the phase-centre SHIFT, and a centred facet has no shift.
     *
     * DDF_OFFSET=<pixels> moves the phase centre, making the w-term active
     * (what happens to every off-centre facet in a faceted imaging run).
     * DDF_NOW=1 zeroes the w-term, enabling a side-by-side comparison.     */
    double off_pix = 0.0;
    if (const char* e = std::getenv("DDF_OFFSET")) off_pix = std::atof(e);
    const double l0  = off_pix * cell;
    const double m0  = off_pix * cell;
    const double lm2 = l0 * l0 + m0 * m0;
    const bool   use_w = (std::getenv("DDF_NOW") == nullptr);
    const double n0m1 = (use_w && lm2 < 1.0) ? (std::sqrt(1.0 - lm2) - 1.0) : 0.0;
    /* -- DDE: direction-dependent complex gain -------------------------------
     * Scalar form of the RIME/Jones formalism. Applied as xG in the degrid
     * and xconj(G) in the adjoint (inside the kernel), which preserves the
     * adjoint relation. Identity (1,0) by default, i.e. no effect.
     * DDF_DDE=1 enables a DETERMINISTIC G, a function of (l0,m0) alone: it
     * must be deterministic, otherwise each node would generate a different
     * value and the result would depend on how many nodes ran.            */
    double gain_re = 1.0, gain_im = 0.0;
    const bool use_dde = (std::getenv("DDF_DDE") != nullptr);
    if (use_dde) {
        const double r   = std::sqrt(lm2);
        const double amp = 1.0 / (1.0 + 40.0 * r * r);   /* primary-beam analogue    */
        const double pha = 150.0 * r;                     /* phase-error analogue     */
        gain_re = amp * std::cos(pha);
        gain_im = amp * std::sin(pha);
    }

    std::printf("  phase  : offset=%.0f px  l0=%.3e m0=%.3e  w-term n0-1=%.6e %s\n",
                off_pix, l0, m0, n0m1, use_w ? "" : "[DDF_NOW: w-term OFF]");
    std::printf("  DDE    : |G|=%.6f  arg(G)=%.6f rad %s\n",
                std::sqrt(gain_re * gain_re + gain_im * gain_im),
                std::atan2(gain_im, gain_re),
                use_dde ? "" : "[identity - set DDF_DDE=1]");

    /* -- PSF (dirty beam): grid of unit visibilities, on the host ---------- */
    std::vector<float> psf_re(ng, 0.0f), psf_im(ng, 0.0f);
    {
        std::vector<float> um_re, um_im;
        std::vector<float> zmdl_re(ng, 0.0f), zmdl_im(ng, 0.0f);
        for (int c = 0; c < nchan; ++c) {
            const ChannelVis& k = channels[c];
            um_re.assign(k.nvis, 1.0f); um_im.assign(k.nvis, 0.0f);
            /* zero model -> dv = v = 1 -> grid(1) = PSF */
            ompc_degrid_residual_grid(k.u.data(), k.v.data(), k.w.data(),
                                      um_re.data(), um_im.data(), k.flag.data(), k.nvis,
                                      zmdl_re.data(), zmdl_im.data(),
                                      psf_re.data(), psf_im.data(),
                                      nx, ny, cell, l0, m0, n0m1,
                                      gain_re, gain_im, KERNEL_W, KERNEL_SG);
        }
        fftshift(psf_re, nx, ny); fftshift(psf_im, nx, ny);
        fft2d(psf_re, psf_im, nx, ny, +1);
        fftshift(psf_re, nx, ny); fftshift(psf_im, nx, ny);
    }
    float beam_peak = 0.0f;
    for (size_t i = 0; i < ng; ++i) if (psf_re[i] > beam_peak) beam_peak = psf_re[i];
    if (beam_peak <= 0.0f) beam_peak = 1.0f;
    std::printf("  PSF    : peak S = %.6g\n", (double)beam_peak);

    /* -- Pipeline state ---------------------------------------------------- */
    std::vector<float> model(ng, 0.0f);                 /* model image (real)   */
    std::vector<float> dirty0;                            /* dirty of cycle 0     */
    std::vector<std::vector<float> > GR(nchan), GI(nchan);
    for (int c = 0; c < nchan; ++c) { GR[c].assign(ng, 0.0f); GI[c].assign(ng, 0.0f); }

    const double t_ini = omp_get_wtime();
    double t_offload = 0.0;

    for (int cycle = 0; cycle < ncycles; ++cycle) {
        std::printf("\n=== MAJOR CYCLE %d ===\n", cycle);

        /* (a) HOST: FFT of the model -> model UV grid */
        std::vector<float> mdl_re(model), mdl_im(ng, 0.0f);
        fftshift(mdl_re, nx, ny); fftshift(mdl_im, nx, ny);
        fft2d(mdl_re, mdl_im, nx, ny, -1);
        fftshift(mdl_re, nx, ny); fftshift(mdl_im, nx, ny);

        for (int c = 0; c < nchan; ++c) {
            std::fill(GR[c].begin(), GR[c].end(), 0.0f);
            std::fill(GI[c].begin(), GI[c].end(), 0.0f);
        }

        /* (b) OFFLOAD: one `target` task per channel -> distinct nodes */
        const double t0 = omp_get_wtime();
        #pragma omp parallel
        #pragma omp single
        {
            for (int c = 0; c < nchan; ++c) {
                const int      nv  = channels[c].nvis;
                const double*  up  = channels[c].u.data();
                const double*  vp  = channels[c].v.data();
                const double*  wp  = channels[c].w.data();
                const float*   vrp = channels[c].re.data();
                const float*   vip = channels[c].im.data();
                const unsigned char* fp = channels[c].flag.data();
                const float*   mrp = mdl_re.data();
                const float*   mip = mdl_im.data();
                float*         grp = GR[c].data();
                float*         gip = GI[c].data();
                const int      ngi = (int)ng;

                /* Canonical OpenMP Cluster form: `target nowait` + `depend`.
                 * This is how the OMPC runtime recognises the regions as
                 * independent tasks and dispatches them to distinct nodes: the
                 * `depend(out:)` on the output grid declares the data
                 * dependency it schedules on.
                 *
                 * Note on local runs: g++ has no offload device, so it executes
                 * the region inline and the tasks end up sequential. That is a
                 * host-fallback limitation (`nowait` PERMITS deferred execution,
                 * it does not require it), not a property of this code - under
                 * OMPC each channel goes to a node. */
                #pragma omp target nowait                                       \
                    depend(out: grp[0:ngi])                                     \
                    map(to: up[0:nv], vp[0:nv], wp[0:nv],                       \
                            vrp[0:nv], vip[0:nv], fp[0:nv],                     \
                            mrp[0:ngi], mip[0:ngi])                             \
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

        /* (c) HOST: sum the per-channel grids (the all-reduce role) */
        std::vector<float> sre(ng, 0.0f), sim(ng, 0.0f);
        for (int c = 0; c < nchan; ++c)
            for (size_t i = 0; i < ng; ++i) { sre[i] += GR[c][i]; sim[i] += GI[c][i]; }

        /* (d) HOST: inverse FFT -> dirty residual image */
        fftshift(sre, nx, ny); fftshift(sim, nx, ny);
        fft2d(sre, sim, nx, ny, +1);
        fftshift(sre, nx, ny); fftshift(sim, nx, ny);

        std::vector<float> dirty(ng);
        for (size_t i = 0; i < ng; ++i) dirty[i] = sre[i] / beam_peak;
        if (cycle == 0) dirty0 = dirty;

        double dmin = dirty[0], dmax = dirty[0], dsum = 0.0;
        for (size_t i = 0; i < ng; ++i) {
            if (dirty[i] < dmin) dmin = dirty[i];
            if (dirty[i] > dmax) dmax = dirty[i];
            dsum += dirty[i];
        }
        std::printf("  dirty  : min=%.6g max=%.6g sum=%.6g\n", dmin, dmax, dsum);

        /* (e) HOST: Hogbom CLEAN - subtract the beam at the peak, N times */
        const int    nminor = 200;
        const double ganho  = 0.1;
        int    pkx = 0, pky = 0;
        for (int it = 0; it < nminor; ++it) {
            float pv = 0.0f;
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const float a = dirty[(size_t)j*nx+i];
                    if (fabsf(a) > fabsf(pv)) { pv = a; pkx = i; pky = j; }
                }
            if (fabsf(pv) < 1e-9f) break;
            const float delta = (float)(ganho * pv);
            model[(size_t)pky*nx+pkx] += delta;
            /* subtract the PSF centred on the peak */
            const int cx = nx/2, cy = ny/2;
            for (int j = 0; j < ny; ++j) {
                const int pj = j - pky + cy;
                if (pj < 0 || pj >= ny) continue;
                for (int i = 0; i < nx; ++i) {
                    const int pi = i - pkx + cx;
                    if (pi < 0 || pi >= nx) continue;
                    dirty[(size_t)j*nx+i] -= delta * psf_re[(size_t)pj*nx+pi] / beam_peak;
                }
            }
        }
        double msum = 0.0; float mmax = 0.0f;
        for (size_t i = 0; i < ng; ++i) { msum += model[i]; if (model[i]>mmax) mmax=model[i]; }
        std::printf("  model  : sum=%.6g peak=%.6g\n", msum, (double)mmax);
    }

    const double t_tot = omp_get_wtime() - t_ini;

    /* -- Output ------------------------------------------------------------ */
    write_fits("dirty_ompc.fits", dirty0, nx, ny);
    write_fits("model_ompc.fits", model, nx, ny);

    /* checksum: invariant - must not depend on the number of nodes */
    double chk = 0.0;
    for (size_t i = 0; i < ng; ++i) chk += fabs((double)dirty0[i]);

    std::printf("\n--------------------------------------------------------\n");
    std::printf("  total time           : %.3f s\n", t_tot);
    std::printf("  time in tasks        : %.3f s (%.1f%%)\n",
                t_offload, 100.0 * t_offload / (t_tot > 0 ? t_tot : 1));
    std::printf("  checksum |dirty|     : %.6f\n", chk);
    std::printf("  FITS: dirty_ompc.fits, model_ompc.fits\n");
    std::printf("--------------------------------------------------------\n");
    std::printf("  INVARIANT: the checksum must be IDENTICAL with 1 node and\n");
    std::printf("  with N nodes. That is the proof the distribution is correct.\n");
    return 0;
}

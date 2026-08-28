/**
 * @file main_ompc.cpp
 * @brief DDFacet (Algoritmo 1) sobre OpenMP Cluster — ciclo completo.
 *
 * ─── O que este programa faz ─────────────────────────────────────────────────
 * Roda os ciclos maiores do DDFacet distribuindo o trabalho POR CANAL entre os
 * nós do cluster, via `#pragma omp target`. Cada canal é uma unidade de trabalho
 * independente (frequência própria) — o desenho "num_channel = 1 per archi.node".
 *
 *   HOST                    TARGET (um por canal, em nós distintos)      HOST
 *   ────                    ────────────────────────────────────────     ────
 *   FFT(modelo) ──map(to)──► degrid → residual → grid ──map(from)──► Σ canais
 *                                                                    → FFT⁻¹
 *                                                                    → CLEAN
 *                                                                    ↺ próximo ciclo
 *
 * ─── Duas restrições do container OMPC que moldaram o projeto ────────────────
 * 1. NÃO há casacore. Por isso a leitura do MS é feita fora, por `tools/ms_export`,
 *    que gera um arquivo .vis por canal (POD puro). Aqui só se faz fread.
 * 2. NÃO se pode contar com FFTW. Por isso a FFT é própria (radix-2 iterativa),
 *    sem dependência externa. Exige dimensões potência de 2.
 *
 * ─── Uso ─────────────────────────────────────────────────────────────────────
 *    ./ddfacet_ompc <prefixo_vis> [npix] [ciclos] [nchan]
 *    ex.: ./ddfacet_ompc data/large 256 5 4
 *         lê data/large_ch0.vis .. _ch3.vis, imagem 256², 5 ciclos maiores
 *
 * Saída: dirty_ompc.fits (imagem suja) e model_ompc.fits (modelo CLEAN).
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

/* ─── Parâmetros do kernel de gridding (iguais ao pipeline principal) ──────── */
static const int    KERNEL_W  = 5;
static const double KERNEL_SG = 1.0;
static const double ARCSEC    = 4.8481368110953599e-06;

/* ═══════════════════════ 1. Leitura dos arquivos .vis ═══════════════════════ */

struct CanalVis {
    int    channel = 0;
    double freq_hz = 0.0;
    double umax_wl = 0.0;
    int    nvis    = 0;
    std::vector<double>        u, v, w;
    std::vector<float>         re, im;
    std::vector<unsigned char> flag;
};

/** @brief Carrega um .vis gerado por tools/ms_export (ver formato lá). */
static bool carregar_vis(const std::string& fn, CanalVis& c) {
    std::FILE* f = std::fopen(fn.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "[erro] nao abri '%s'\n", fn.c_str()); return false; }

    char magic[8] = {0};
    std::int32_t nvis = 0, ch = 0;
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "DDFVIS01", 8) != 0) {
        std::fprintf(stderr, "[erro] '%s' nao e um arquivo DDFVIS01\n", fn.c_str());
        std::fclose(f); return false;
    }
    /* Um .vis truncado daria lixo silencioso — cada leitura é conferida. */
    std::size_t lidos = 0;
    lidos += std::fread(&nvis,      sizeof(std::int32_t), 1, f);
    lidos += std::fread(&ch,        sizeof(std::int32_t), 1, f);
    lidos += std::fread(&c.freq_hz, sizeof(double),       1, f);
    lidos += std::fread(&c.umax_wl, sizeof(double),       1, f);
    if (lidos != 4 || nvis <= 0) {
        std::fprintf(stderr, "[erro] cabecalho invalido em '%s' (nvis=%d)\n", fn.c_str(), nvis);
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
        std::fprintf(stderr, "[erro] '%s' truncado: li %zu de %zu elementos\n",
                     fn.c_str(), got, 6 * n);
    std::fclose(f);
    return ok;
}

/* ═══════════════════════ 2. FFT própria (sem FFTW) ═════════════════════════ */

/** @brief FFT 1-D radix-2 iterativa, in-place. sinal=-1 direta, +1 inversa. */
static void fft1d(float* re, float* im, int n, int sinal) {
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
        const double ang = sinal * 2.0 * OMPC_PI / len;
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
    if (sinal > 0) for (int i = 0; i < n; ++i) { re[i] /= n; im[i] /= n; }
}

/** @brief FFT 2-D separável (linhas depois colunas). n deve ser potência de 2. */
static void fft2d(std::vector<float>& re, std::vector<float>& im, int nx, int ny, int sinal) {
    std::vector<float> lr(nx), li(nx);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) { lr[i] = re[(size_t)j*nx+i]; li[i] = im[(size_t)j*nx+i]; }
        fft1d(lr.data(), li.data(), nx, sinal);
        for (int i = 0; i < nx; ++i) { re[(size_t)j*nx+i] = lr[i]; im[(size_t)j*nx+i] = li[i]; }
    }
    std::vector<float> cr(ny), ci(ny);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) { cr[j] = re[(size_t)j*nx+i]; ci[j] = im[(size_t)j*nx+i]; }
        fft1d(cr.data(), ci.data(), ny, sinal);
        for (int j = 0; j < ny; ++j) { re[(size_t)j*nx+i] = cr[j]; im[(size_t)j*nx+i] = ci[j]; }
    }
}

/** @brief Troca os quadrantes (fftshift), levando a origem ao centro e vice-versa. */
static void fftshift(std::vector<float>& a, int nx, int ny) {
    std::vector<float> t(a.size());
    const int hx = nx / 2, hy = ny / 2;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            t[(size_t)((j + hy) % ny) * nx + (i + hx) % nx] = a[(size_t)j * nx + i];
    a.swap(t);
}

/* ═══════════════════════ 3. Escrita FITS (sem dependências) ════════════════ */

static bool escrever_fits(const char* fn, const std::vector<float>& img, int nx, int ny) {
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

/* ═══════════════════════════════ 4. main ═══════════════════════════════════ */

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "uso: %s <prefixo_vis> [npix] [ciclos] [nchan]\n"
            "  ex: %s data/large 256 5 4\n", argv[0], argv[0]);
        return 1;
    }
    const std::string prefixo = argv[1];
    int npix   = (argc > 2) ? std::atoi(argv[2]) : 256;
    int nciclo = (argc > 3) ? std::atoi(argv[3]) : 5;
    int nchan  = (argc > 4) ? std::atoi(argv[4]) : 4;

    /* a FFT radix-2 exige potência de 2 */
    if (npix & (npix - 1)) {
        std::fprintf(stderr, "[erro] npix deve ser potencia de 2 (recebi %d)\n", npix);
        return 1;
    }

    std::printf("========================================================\n");
    std::printf("  DDFacet OMPC — ciclo completo, distribuido por CANAL\n");
    std::printf("  [HEAD] pid=%d\n", getpid());
    std::printf("========================================================\n");

    /* ── Carrega um .vis por canal ─────────────────────────────────────────── */
    std::vector<CanalVis> canais(nchan);
    double umax_global = 0.0;
    for (int c = 0; c < nchan; ++c) {
        char fn[1024];
        std::snprintf(fn, sizeof(fn), "%s_ch%d.vis", prefixo.c_str(), c);
        if (!carregar_vis(fn, canais[c])) return 1;
        if (canais[c].umax_wl > umax_global) umax_global = canais[c].umax_wl;
        std::printf("  canal %d: %8d vis | %.3f MHz | u_max %.1f lambda\n",
                    c, canais[c].nvis, canais[c].freq_hz / 1e6, canais[c].umax_wl);
    }

    /* cell derivado do maior u_max (amostragem de Nyquist com folga 3x) */
    const double cell = 1.0 / (3.0 * umax_global);
    const int    nx = npix, ny = npix;
    const size_t ng = (size_t)nx * ny;
    std::printf("  imagem : %dx%d  cell=%.4f arcsec  ciclos=%d\n",
                nx, ny, cell / ARCSEC, nciclo);

    /* ── Centro de fase da faceta ─────────────────────────────────────────────
     * Por padrão a faceta fica na origem (l0=m0=0). Nesse caso n0 = 1 e o
     * TERMO-W se anula — por construção, não por omissão: o termo w·(n0−1)
     * aparece no DESLOCAMENTO de centro de fase, e uma faceta centrada não tem
     * deslocamento nenhum.
     *
     * DDF_OFFSET=<pixels> desloca o centro de fase, tornando o termo-w atuante
     * (é o que acontece em cada faceta fora do centro num imageamento faceteado).
     * DDF_NOW=1 zera o termo-w, permitindo o comparativo lado a lado.        */
    double off_pix = 0.0;
    if (const char* e = std::getenv("DDF_OFFSET")) off_pix = std::atof(e);
    const double l0  = off_pix * cell;
    const double m0  = off_pix * cell;
    const double lm2 = l0 * l0 + m0 * m0;
    const bool   use_w = (std::getenv("DDF_NOW") == nullptr);
    const double n0m1 = (use_w && lm2 < 1.0) ? (std::sqrt(1.0 - lm2) - 1.0) : 0.0;
    const double gain_re = 1.0, gain_im = 0.0;

    std::printf("  fase   : offset=%.0f px  l0=%.3e m0=%.3e  termo-w n0-1=%.6e %s\n",
                off_pix, l0, m0, n0m1, use_w ? "" : "[DDF_NOW: DESLIGADO]");

    /* ── PSF (dirty beam): grade das visibilidades unitárias, no host ──────── */
    std::vector<float> psf_re(ng, 0.0f), psf_im(ng, 0.0f);
    {
        std::vector<float> um_re, um_im;
        std::vector<float> zmdl_re(ng, 0.0f), zmdl_im(ng, 0.0f);
        for (int c = 0; c < nchan; ++c) {
            const CanalVis& k = canais[c];
            um_re.assign(k.nvis, 1.0f); um_im.assign(k.nvis, 0.0f);
            /* modelo zerado → δv = v = 1 → grid(1) = PSF */
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
    std::printf("  PSF    : pico S = %.6g\n", (double)beam_peak);

    /* ── Estado do pipeline ────────────────────────────────────────────────── */
    std::vector<float> modelo(ng, 0.0f);                 /* imagem modelo (real) */
    std::vector<float> dirty0;                            /* dirty do 1o ciclo   */
    std::vector<std::vector<float> > GR(nchan), GI(nchan);
    for (int c = 0; c < nchan; ++c) { GR[c].assign(ng, 0.0f); GI[c].assign(ng, 0.0f); }

    const double t_ini = omp_get_wtime();
    double t_offload = 0.0;

    for (int ciclo = 0; ciclo < nciclo; ++ciclo) {
        std::printf("\n=== CICLO MAIOR %d ===\n", ciclo);

        /* (a) HOST: FFT do modelo -> grade UV do modelo */
        std::vector<float> mdl_re(modelo), mdl_im(ng, 0.0f);
        fftshift(mdl_re, nx, ny); fftshift(mdl_im, nx, ny);
        fft2d(mdl_re, mdl_im, nx, ny, -1);
        fftshift(mdl_re, nx, ny); fftshift(mdl_im, nx, ny);

        for (int c = 0; c < nchan; ++c) {
            std::fill(GR[c].begin(), GR[c].end(), 0.0f);
            std::fill(GI[c].begin(), GI[c].end(), 0.0f);
        }

        /* (b) OFFLOAD: uma tarefa `target` por canal → nós distintos */
        const double t0 = omp_get_wtime();
        #pragma omp parallel
        #pragma omp single
        {
            for (int c = 0; c < nchan; ++c) {
                const int      nv  = canais[c].nvis;
                const double*  up  = canais[c].u.data();
                const double*  vp  = canais[c].v.data();
                const double*  wp  = canais[c].w.data();
                const float*   vrp = canais[c].re.data();
                const float*   vip = canais[c].im.data();
                const unsigned char* fp = canais[c].flag.data();
                const float*   mrp = mdl_re.data();
                const float*   mip = mdl_im.data();
                float*         grp = GR[c].data();
                float*         gip = GI[c].data();
                const int      ngi = (int)ng;

                /* Forma canônica do OpenMP Cluster: `target nowait` + `depend`.
                 * É assim que o runtime do OMPC identifica as regiões como
                 * tarefas independentes e as despacha para nós distintos — o
                 * `depend(out:)` sobre a grade de saída declara a dependência de
                 * dados que ele usa para o escalonamento.
                 *
                 * Nota sobre execução local: o g++, sem device de offload,
                 * executa a região inline e as tarefas saem sequenciais. Isso é
                 * limitação do fallback de host (o `nowait` PERMITE execução
                 * diferida, não obriga), não do código — sob OMPC cada canal vai
                 * para um nó. Para medir paralelismo na máquina local, use o
                 * caminho MPI (build/ddfacet_mpi). */
                #pragma omp target nowait                                       \
                    depend(out: grp[0:ngi])                                     \
                    map(to: up[0:nv], vp[0:nv], wp[0:nv],                       \
                            vrp[0:nv], vip[0:nv], fp[0:nv],                     \
                            mrp[0:ngi], mip[0:ngi])                             \
                    map(tofrom: grp[0:ngi], gip[0:ngi])
                {
                    printf("[WORKER] ciclo %d canal %d  pid=%d\n", ciclo, c, getpid());
                    ompc_degrid_residual_grid(up, vp, wp, vrp, vip, fp, nv,
                                              mrp, mip, grp, gip,
                                              nx, ny, cell, l0, m0, n0m1,
                                              gain_re, gain_im, KERNEL_W, KERNEL_SG);
                }
            }
            #pragma omp taskwait
        }
        t_offload += omp_get_wtime() - t0;

        /* (c) HOST: soma as grades dos canais (o papel do all-reduce) */
        std::vector<float> sre(ng, 0.0f), sim(ng, 0.0f);
        for (int c = 0; c < nchan; ++c)
            for (size_t i = 0; i < ng; ++i) { sre[i] += GR[c][i]; sim[i] += GI[c][i]; }

        /* (d) HOST: FFT⁻¹ -> imagem residual suja */
        fftshift(sre, nx, ny); fftshift(sim, nx, ny);
        fft2d(sre, sim, nx, ny, +1);
        fftshift(sre, nx, ny); fftshift(sim, nx, ny);

        std::vector<float> dirty(ng);
        for (size_t i = 0; i < ng; ++i) dirty[i] = sre[i] / beam_peak;
        if (ciclo == 0) dirty0 = dirty;

        double dmin = dirty[0], dmax = dirty[0], dsum = 0.0;
        for (size_t i = 0; i < ng; ++i) {
            if (dirty[i] < dmin) dmin = dirty[i];
            if (dirty[i] > dmax) dmax = dirty[i];
            dsum += dirty[i];
        }
        std::printf("  dirty  : min=%.6g max=%.6g soma=%.6g\n", dmin, dmax, dsum);

        /* (e) HOST: CLEAN de Högbom — subtrai o beam no pico, N vezes */
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
            modelo[(size_t)pky*nx+pkx] += delta;
            /* subtrai a PSF centrada no pico */
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
        for (size_t i = 0; i < ng; ++i) { msum += modelo[i]; if (modelo[i]>mmax) mmax=modelo[i]; }
        std::printf("  modelo : soma=%.6g pico=%.6g\n", msum, (double)mmax);
    }

    const double t_tot = omp_get_wtime() - t_ini;

    /* ── Saída ─────────────────────────────────────────────────────────────── */
    escrever_fits("dirty_ompc.fits", dirty0, nx, ny);
    escrever_fits("model_ompc.fits", modelo, nx, ny);

    /* checksum: invariante — nao pode depender do numero de nos */
    double chk = 0.0;
    for (size_t i = 0; i < ng; ++i) chk += fabs((double)dirty0[i]);

    std::printf("\n--------------------------------------------------------\n");
    std::printf("  tempo total          : %.3f s\n", t_tot);
    std::printf("  tempo nas tarefas    : %.3f s (%.1f%%)\n",
                t_offload, 100.0 * t_offload / (t_tot > 0 ? t_tot : 1));
    std::printf("  checksum |dirty|     : %.6f\n", chk);
    std::printf("  FITS: dirty_ompc.fits, model_ompc.fits\n");
    std::printf("--------------------------------------------------------\n");
    std::printf("  INVARIANTE: o checksum tem de ser IDENTICO com 1 no e\n");
    std::printf("  com N nos. E a prova de que a distribuicao esta correta.\n");
    return 0;
}

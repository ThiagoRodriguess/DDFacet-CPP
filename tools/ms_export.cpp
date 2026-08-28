/**
 * @file ms_export.cpp
 * @brief Export a Measurement Set to simple binary files, ONE PER CHANNEL.
 *
 * --- Why this tool exists ---------------------------------------------------
 * The OpenMP Cluster container ships the patched clang and the offload runtime,
 * but NOT casacore. An OMPC binary linking casacore will not build in it.
 *
 * The natural split is:
 *
 *    ms_export   (HOST, g++ + casacore)   MS  --->  .vis per channel
 *    ddfacet_ompc(CONTAINER, clang)       .vis --->  offload per channel ---> image
 *
 * Beyond solving the dependency, this is good design: the MS is parsed ONCE and
 * repeated distributed runs read a cheap flat format, with no casacore table
 * parsing on every node.
 *
 * --- .vis file format (little-endian, plain POD) ----------------------------
 *    [header]
 *      char   magic[8]  = "DDFVIS01"
 *      int32  nvis
 *      int32  channel
 *      double freq_hz          frequency of THIS channel
 *      double umax_wl          max|(u,v)| in wavelengths (to derive cell_size)
 *    [data, in this order]
 *      double u[nvis], v[nvis], w[nvis]     in wavelengths
 *      float  re[nvis], im[nvis]            measured visibility
 *      uint8  flag[nvis]                    1 = discard
 *
 * Arrays are separate (not interleaved) because that is how the OMPC kernel
 * consumes them: each becomes its own map() clause.
 *
 * --- Usage ------------------------------------------------------------------
 *    ./ms_export <path.ms> <output_prefix> [max_rows]
 *    e.g.: ./ms_export ../data/sim_large.ms/sim_large.ms data/large
 *         -> data/large_ch0.vis, data/large_ch1.vis, ...
 *
 * Build:
 *    g++ -std=c++17 -O2 -Iinclude -isystem /usr/include/casacore \
 *        tools/ms_export.cpp src/ms_io.cpp \
 *        -lcasa_ms -lcasa_tables -lcasa_casa -o build/ms_export
 */
#include "ms_io.h"

#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/casa/Arrays/Array.h>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

using namespace ddfacet;

/** @brief How many channels the MS has (SPECTRAL_WINDOW subtable). */
static int contar_canais(const std::string& ms_path) {
    try {
        casacore::Table spw(ms_path + "/SPECTRAL_WINDOW");
        casacore::ArrayColumn<casacore::Double> cf(spw, "CHAN_FREQ");
        return static_cast<int>(cf.get(0).shape()(0));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ms_export] could not read SPECTRAL_WINDOW: %s\n", e.what());
        return -1;
    }
}

/** @brief Write a VisibilitySet in the .vis format described above. */
static bool gravar_vis(const std::string& fn, const VisibilitySet& vis,
                       int channel, double freq_hz, double umax_wl) {
    std::FILE* f = std::fopen(fn.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "[ms_export] could not open '%s'\n", fn.c_str()); return false; }

    const std::int32_t nvis = static_cast<std::int32_t>(vis.nvis);
    const std::int32_t ch   = static_cast<std::int32_t>(channel);

    std::fwrite("DDFVIS01", 1, 8, f);
    std::fwrite(&nvis,    sizeof(std::int32_t), 1, f);
    std::fwrite(&ch,      sizeof(std::int32_t), 1, f);
    std::fwrite(&freq_hz, sizeof(double),       1, f);
    std::fwrite(&umax_wl, sizeof(double),       1, f);

    /* u, v, w already come in wavelengths from read_ms */
    std::fwrite(vis.u.data(), sizeof(double), vis.nvis, f);
    std::fwrite(vis.v.data(), sizeof(double), vis.nvis, f);
    std::fwrite(vis.w.data(), sizeof(double), vis.nvis, f);

    /* complex -> two arrays (re/im), what the offloadable kernel consumes */
    std::vector<float> re(vis.nvis), im(vis.nvis);
    std::vector<std::uint8_t> fl(vis.nvis);
    for (std::size_t k = 0; k < vis.nvis; ++k) {
        re[k] = vis.data[k].real();
        im[k] = vis.data[k].imag();
        fl[k] = vis.flag[k] ? 1u : 0u;
    }
    std::fwrite(re.data(), sizeof(float),        vis.nvis, f);
    std::fwrite(im.data(), sizeof(float),        vis.nvis, f);
    std::fwrite(fl.data(), sizeof(std::uint8_t), vis.nvis, f);

    const bool ok = (std::ferror(f) == 0);
    std::fclose(f);
    return ok;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <path.ms> <output_prefix> [max_rows]\n"
            "  e.g. %s ../data/sim_large.ms/sim_large.ms data/large\n"
            "      -> data/large_ch0.vis, data/large_ch1.vis, ...\n",
            argv[0], argv[0]);
        return 1;
    }
    const std::string ms_path = argv[1];
    const std::string prefixo = argv[2];
    const long max_linhas = (argc > 3) ? std::atol(argv[3]) : -1;

    const int nchan = contar_canais(ms_path);
    if (nchan <= 0) return 1;

    /* Channel 0 metadata: u_max suggests the cell size for the pipeline. */
    double freq0 = 0.0, umax0 = 0.0;
    long nrows = 0;
    int nch_check = 0;
    if (!read_ms_metadata(ms_path, freq0, umax0, nrows, 0, &nch_check)) return 1;

    const long nler = (max_linhas > 0 && max_linhas < nrows) ? max_linhas : nrows;

    std::printf("========================================================\n");
    std::printf("  ms_export - MS -> one .vis file per channel\n");
    std::printf("  MS      : %s\n", ms_path.c_str());
    std::printf("  rows    : %ld (of %ld)\n", nler, nrows);
    std::printf("  channels: %d   (each becomes one OMPC unit of work)\n", nchan);
    std::printf("  u_max   : %.1f lambda  -> suggested cell %.4f arcsec\n",
                umax0, (1.0 / (3.0 * umax0)) * 206264.806247);
    std::printf("========================================================\n");

    for (int c = 0; c < nchan; ++c) {
        VisibilitySet vis;
        double freq = 0.0;
        if (!read_ms(ms_path, vis, freq, 0, nler, c)) {
            std::fprintf(stderr, "[ms_export] failed reading channel %d\n", c);
            return 1;
        }
        /* u_max for this channel (it changes with frequency) */
        double umax = 0.0;
        for (std::size_t k = 0; k < vis.nvis; ++k) {
            const double r = std::sqrt(vis.u[k] * vis.u[k] + vis.v[k] * vis.v[k]);
            if (r > umax) umax = r;
        }

        char fn[1024];
        std::snprintf(fn, sizeof(fn), "%s_ch%d.vis", prefixo.c_str(), c);
        if (!gravar_vis(fn, vis, c, freq, umax)) return 1;

        std::printf("  channel %d: %8zu vis | %.3f MHz | u_max %.1f lambda -> %s\n",
                    c, vis.nvis, freq / 1e6, umax, fn);
    }

    std::printf("--------------------------------------------------------\n");
    std::printf("  Done. Copy the .vis files to the cluster with the OMPC binary.\n");
    return 0;
}

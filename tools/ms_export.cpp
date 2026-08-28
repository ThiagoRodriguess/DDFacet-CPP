/**
 * @file ms_export.cpp
 * @brief Export a Measurement Set to .vis files, one per spectral channel.
 *
 * This is the host half of the pipeline. It parses the MS with casacore and
 * writes the flat POD format defined in vis_file.h, which the imaging binary
 * reads without any external library — necessary because the OpenMP Cluster
 * container ships the patched compiler but not casacore.
 *
 * Beyond satisfying that constraint it is also cheaper: the MS is parsed once,
 * and repeated distributed runs read a format that needs no parsing at all.
 *
 * Usage:
 *   ./ms_export <path.ms> <output_prefix> [max_rows]
 *   e.g. ./ms_export data/sim_large.ms data/obs
 *        -> data/obs_ch0.vis, data/obs_ch1.vis, ...
 */
#include "ms_io.h"
#include "vis_file.h"

#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/casa/Arrays/Array.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace ddfacet;

namespace {

constexpr double RAD_TO_ARCSEC = 206264.806247;

/** @brief How many spectral channels the MS has. */
int count_channels(const std::string& ms_path) {
    try {
        casacore::Table spw(ms_path + "/SPECTRAL_WINDOW");
        casacore::ArrayColumn<casacore::Double> cf(spw, "CHAN_FREQ");
        return static_cast<int>(cf.get(0).shape()(0));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[ms_export] could not read SPECTRAL_WINDOW: %s\n", e.what());
        return -1;
    }
}

/** @brief Convert a VisibilitySet into the on-disk representation. */
VisFile to_vis_file(const VisibilitySet& vis, int channel, double freq_hz) {
    VisFile out;
    out.channel = channel;
    out.freq_hz = freq_hz;
    out.resize(static_cast<int>(vis.nvis));

    double umax = 0.0;
    for (std::size_t k = 0; k < vis.nvis; ++k) {
        out.u[k] = vis.u[k];
        out.v[k] = vis.v[k];
        out.w[k] = vis.w[k];
        out.re[k] = vis.data[k].real();
        out.im[k] = vis.data[k].imag();
        out.flag[k] = vis.flag[k] ? 1u : 0u;

        const double r = std::sqrt(vis.u[k] * vis.u[k] + vis.v[k] * vis.v[k]);
        if (r > umax) umax = r;
    }
    out.umax_wl = umax;
    return out;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <path.ms> <output_prefix> [max_rows]\n"
            "  e.g. %s data/sim_large.ms data/obs\n"
            "       -> data/obs_ch0.vis, data/obs_ch1.vis, ...\n",
            argv[0], argv[0]);
        return 1;
    }
    const std::string ms_path  = argv[1];
    const std::string prefix   = argv[2];
    const long        max_rows = (argc > 3) ? std::atol(argv[3]) : -1;

    const int nchan = count_channels(ms_path);
    if (nchan <= 0) return 1;

    // Channel 0 metadata: u_max suggests the cell size for the pipeline.
    double freq0 = 0.0, umax0 = 0.0;
    long   nrows = 0;
    if (!read_ms_metadata(ms_path, freq0, umax0, nrows)) return 1;

    const long rows_to_read =
        (max_rows > 0 && max_rows < nrows) ? max_rows : nrows;

    std::printf("========================================================\n");
    std::printf("  ms_export - MS -> one .vis file per channel\n");
    std::printf("  MS      : %s\n", ms_path.c_str());
    std::printf("  rows    : %ld (of %ld)\n", rows_to_read, nrows);
    std::printf("  channels: %d   (each becomes one unit of work)\n", nchan);
    std::printf("  u_max   : %.1f lambda  -> suggested cell %.4f arcsec\n",
                umax0, (1.0 / (3.0 * umax0)) * RAD_TO_ARCSEC);
    std::printf("========================================================\n");

    for (int c = 0; c < nchan; ++c) {
        VisibilitySet vis;
        double freq = 0.0;
        if (!read_ms(ms_path, vis, freq, 0, rows_to_read, c)) {
            std::fprintf(stderr, "[ms_export] failed reading channel %d\n", c);
            return 1;
        }

        const VisFile out = to_vis_file(vis, c, freq);

        char fn[1024];
        std::snprintf(fn, sizeof(fn), "%s_ch%d.vis", prefix.c_str(), c);
        if (!vis_file_write(fn, out)) return 1;

        std::printf("  channel %d: %8d vis | %.3f MHz | u_max %.1f lambda -> %s\n",
                    c, out.nvis, out.freq_hz / 1e6, out.umax_wl, fn);
    }

    std::printf("--------------------------------------------------------\n");
    std::printf("  Done. Copy the .vis files to the cluster with the binary.\n");
    return 0;
}

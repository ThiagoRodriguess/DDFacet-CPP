/**
 * @file vis_file.h
 * @brief The .vis interchange format: one channel of visibilities, plain POD.
 *
 * This is the boundary between the two halves of the pipeline. `ms_export`
 * parses a Measurement Set with casacore and writes these files; the imaging
 * binary reads them and knows nothing about casacore. That split exists
 * because the OpenMP Cluster container ships the patched compiler but not the
 * scientific libraries.
 *
 * Defining reader and writer here, together, keeps the two ends from drifting
 * apart — a mismatch would be silent, since the payload is raw bytes.
 *
 * Layout (little-endian):
 *
 *   [header]
 *     char   magic[8]  = "DDFVIS01"
 *     int32  nvis
 *     int32  channel
 *     double freq_hz          frequency of this channel
 *     double umax_wl          max |(u,v)| in wavelengths
 *   [payload, in this order]
 *     double u[nvis], v[nvis], w[nvis]     in wavelengths
 *     float  re[nvis], im[nvis]            measured visibility
 *     uint8  flag[nvis]                    1 = discard
 *
 * The arrays are stored separately rather than interleaved because that is how
 * the offloaded kernel consumes them: each one becomes its own map() clause.
 */
#ifndef VIS_FILE_H
#define VIS_FILE_H

#include <cstddef>
#include <string>
#include <vector>

namespace ddfacet {

/** @brief One channel's visibilities, as they live on disk. */
struct VisFile {
    int    channel = 0;
    double freq_hz = 0.0;
    double umax_wl = 0.0;
    int    nvis    = 0;

    std::vector<double>        u, v, w;   ///< coordinates [wavelengths]
    std::vector<float>         re, im;    ///< measured visibility
    std::vector<unsigned char> flag;      ///< 1 = discard

    /** @brief Resize every array to hold n visibilities. */
    void resize(int n);
};

/**
 * @brief Read a .vis file.
 *
 * Every read is checked: a truncated file would otherwise be interpreted as
 * valid data and produce silently wrong images.
 *
 * @return false on a missing file, a bad magic number, or a short read.
 */
bool vis_file_read(const std::string& path, VisFile& out);

/** @brief Write a .vis file. @return false on any I/O error. */
bool vis_file_write(const std::string& path, const VisFile& in);

} // namespace ddfacet

#endif // VIS_FILE_H

/**
 * @file ms_io.h
 * @brief Measurement Set reading (CASA format) through casacore.
 *
 * Isolates the casacore dependency: only ms_io.cpp includes its headers, and
 * only tools/ms_export links against it. The imaging binary never sees it,
 * because the OpenMP Cluster container does not ship casacore.
 */
#ifndef MS_IO_H
#define MS_IO_H

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

namespace ddfacet {

/** Speed of light [m/s] — converts UVW from metres to wavelengths. */
constexpr double C_LIGHT = 299792458.0;

/**
 * @brief One channel's worth of visibilities read from a Measurement Set.
 *
 * The (u, v, w) coordinates are stored already divided by the wavelength, so
 * in wavelengths rather than metres. Since the wavelength depends on the
 * channel, a VisibilitySet always belongs to one specific channel.
 */
struct VisibilitySet {
    std::vector<double> u;                   ///< u coordinate [wavelengths]
    std::vector<double> v;                   ///< v coordinate [wavelengths]
    std::vector<double> w;                   ///< w coordinate [wavelengths]
    std::vector<std::complex<float>> data;   ///< measured visibility
    std::vector<float> weight;               ///< weight
    std::vector<bool> flag;                  ///< true = discard

    std::size_t nvis = 0;                    ///< number of visibilities
    double freq_ref = 0.0;                   ///< channel frequency [Hz]
    double wavelength = 0.0;                 ///< wavelength [m]

    /** @brief Resize every array to hold n visibilities. */
    void resize(std::size_t n) {
        nvis = n;
        u.resize(n);
        v.resize(n);
        w.resize(n);
        data.resize(n);
        weight.resize(n);
        flag.resize(n, false);
    }

    /** @brief Count the visibilities that are not flagged. */
    std::size_t count_valid() const {
        std::size_t count = 0;
        for (std::size_t i = 0; i < nvis; ++i)
            if (!flag[i]) ++count;
        return count;
    }
};

/**
 * @brief Read (a row range of) a Measurement Set into a VisibilitySet.
 *
 * Reads UVW (metres), DATA, FLAG and WEIGHT from the MAIN table, and the
 * channel frequency from SPECTRAL_WINDOW. UVW is converted to wavelengths.
 *
 * The channel is explicit because each channel is an independent unit of work
 * with its own frequency — the axis along which the pipeline is distributed.
 * The frequency used is CHAN_FREQ[channel]: changing channel changes the
 * wavelength, and therefore the UVW conversion.
 *
 * @param path      path to the .ms directory
 * @param vis       output, filled with the visibilities
 * @param freq_out  frequency of the channel read [Hz]
 * @param startrow  first row to read (default 0)
 * @param nrow      number of rows (default -1 = to the end)
 * @param channel   spectral channel index (default 0)
 * @return true on success; false on error, with a message on std::cerr
 */
bool read_ms(const std::string& path, VisibilitySet& vis, double& freq_out,
             long startrow = 0, long nrow = -1, int channel = 0);

/**
 * @brief Read only an MS's metadata, without loading the visibilities.
 *
 * @param path        path to the .ms
 * @param freq_out    channel frequency [Hz]
 * @param umax_wl_out max |(u,v)| in wavelengths, for the requested channel
 * @param nrows_out   total number of rows in the MS
 * @param channel     spectral channel index (default 0)
 * @param nchan_out   optional: total number of channels in the MS
 * @return true on success
 */
bool read_ms_metadata(const std::string& path, double& freq_out,
                      double& umax_wl_out, long& nrows_out,
                      int channel = 0, int* nchan_out = nullptr);

} // namespace ddfacet

#endif // MS_IO_H

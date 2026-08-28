/**
 * @file ddfacet.h
 * @brief Shared types for the imaging pipeline.
 *
 * Kept deliberately small: the OMPC pipeline (src/main_ompc.cpp) is
 * self-contained and works over flat POD arrays, so the only shared type left
 * is the one the Measurement Set reader produces.
 */
#ifndef DDFACET_H
#define DDFACET_H

#include <complex>
#include <cstddef>
#include <vector>

namespace ddfacet {

/** Speed of light [m/s] — converts UVW from metres to wavelengths. */
constexpr double C_LIGHT = 299792458.0;

/**
 * @brief One channel's worth of visibilities read from a Measurement Set.
 *
 * The (u, v, w) coordinates are stored already divided by the wavelength, i.e.
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

    VisibilitySet() = default;

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

} // namespace ddfacet

#endif // DDFACET_H

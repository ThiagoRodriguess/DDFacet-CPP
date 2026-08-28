/**
 * @file ms_io.h
 * @brief Measurement Set reading (CASA format) through casacore.
 *
 * Isolates the casacore dependency: only ms_io.cpp includes its headers, so the
 * rest of the tree stays free of it. This matters because the OMPC container
 * does not ship casacore — MS reading happens on the host, in tools/ms_export.
 */
#ifndef MS_IO_H
#define MS_IO_H

#include "ddfacet.h"
#include <string>

namespace ddfacet {

/**
 * @brief Read (a row range of) a Measurement Set into a VisibilitySet.
 *
 * Reads UVW (metres), DATA, FLAG and WEIGHT from the MAIN table, and the
 * channel frequency from SPECTRAL_WINDOW. UVW is converted from metres to
 * wavelengths on the way out.
 *
 * The channel is explicit because each channel is an independent unit of work
 * with its own frequency — the axis along which the pipeline is distributed.
 * Note that the frequency used is CHAN_FREQ[channel]: changing channel changes
 * the wavelength, and therefore the UVW conversion.
 *
 * The [startrow, startrow + nrow) range allows a large MS to be read in
 * slices rather than all at once.
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
 * @brief Read only an MS's metadata, without loading the visibilities:
 * frequency, max |(u,v)| in wavelengths (to derive the cell size), and the
 * total row count.
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

/**
 * @file ms_io.cpp
 * @brief Measurement Set reading, implemented with casacore.
 *
 * Link with: -lcasa_ms -lcasa_tables -lcasa_casa
 */
#include "ms_io.h"

#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/tables/Tables/RefRows.h>
#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Cube.h>

#include <iostream>
#include <cmath>

namespace ddfacet {

bool read_ms(const std::string& path, VisibilitySet& vis, double& freq_out,
             long startrow, long nrow, int channel) {
    using namespace casacore;
    try {
        Table ms(path);
        const uInt total = static_cast<uInt>(ms.nrow());
        if (total == 0) {
            std::cerr << "[read_ms] empty MS: " << path << "\n";
            return false;
        }

        // Row range [s, s+n): lets a large MS be read in slices.
        uInt s = (startrow < 0) ? 0u : static_cast<uInt>(startrow);
        if (s >= total) s = total - 1;
        uInt n = (nrow < 0) ? (total - s)
                            : std::min(static_cast<uInt>(nrow), total - s);

        // -- Frequency of the requested CHANNEL (CHAN_FREQ[channel]) ---------
        // The channel is the independent unit of work (channel c -> node c).
        // NOTE: each channel has its own frequency, hence its own wavelength,
        // so the UVW [m] -> [wavelengths] conversion below depends on it.
        Table spw(path + "/SPECTRAL_WINDOW");
        ArrayColumn<Double> chanFreq(spw, "CHAN_FREQ");
        Array<Double> cf = chanFreq.get(0);
        const int nchan_spw = static_cast<int>(cf.shape()(0));
        if (channel < 0 || channel >= nchan_spw) {
            std::cerr << "[read_ms] channel " << channel << " out of range [0,"
                      << nchan_spw << ") in '" << path << "'\n";
            return false;
        }
        const double freq = cf(IPosition(1, channel));
        const double wl   = C_LIGHT / freq;        // wavelength [m]

        // -- MAIN table columns, restricted to this row range ----------------
        ArrayColumn<Double>  uvwCol(ms,    "UVW");
        ArrayColumn<Complex> dataCol(ms,   "DATA");
        ArrayColumn<Bool>    flagCol(ms,   "FLAG");
        ArrayColumn<Float>   weightCol(ms, "WEIGHT");

        const RefRows rows(s, s + n - 1);          // inclusive -> n rows
        Matrix<Double>  uvw  = uvwCol.getColumnCells(rows);     // [3, n]
        Cube<Complex>   data = dataCol.getColumnCells(rows);    // [npol, nchan, n]
        Cube<Bool>      flag = flagCol.getColumnCells(rows);    // [npol, nchan, n]
        Matrix<Float>   wgt  = weightCol.getColumnCells(rows);  // [npol, n]

        // The DATA column may carry fewer channels than SPECTRAL_WINDOW says.
        const int nchan_data = static_cast<int>(data.shape()(1));
        if (channel >= nchan_data) {
            std::cerr << "[read_ms] channel " << channel << " missing from DATA ("
                      << nchan_data << " channels) in '" << path << "'\n";
            return false;
        }

        vis.resize(n);
        for (uInt i = 0; i < n; ++i) {
            // UVW: metres -> wavelengths
            vis.u[i] = uvw(0, i) / wl;
            vis.v[i] = uvw(1, i) / wl;
            vis.w[i] = uvw(2, i) / wl;

            // DATA / FLAG: polarisation 0, channel `channel`
            const Complex c = data(0, channel, i);
            vis.data[i]   = std::complex<float>(c.real(), c.imag());
            vis.flag[i]   = flag(0, channel, i);
            vis.weight[i] = wgt(0, i);
        }

        vis.freq_ref   = freq;
        vis.wavelength = wl;
        freq_out       = freq;

        std::cout << "  [read_ms] " << path << ": rows [" << s << ".." << (s + n)
                  << ") = " << n << " visibilities | channel " << channel << "/"
                  << nchan_spw << " | freq " << freq / 1e6 << " MHz\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[read_ms] error reading '" << path << "': " << e.what() << "\n";
        return false;
    }
}

bool read_ms_metadata(const std::string& path, double& freq_out,
                      double& umax_wl_out, long& nrows_out,
                      int channel, int* nchan_out) {
    using namespace casacore;
    try {
        Table ms(path);
        const uInt total = static_cast<uInt>(ms.nrow());

        Table spw(path + "/SPECTRAL_WINDOW");
        ArrayColumn<Double> chanFreq(spw, "CHAN_FREQ");
        Array<Double> cf = chanFreq.get(0);
        const int nchan = static_cast<int>(cf.shape()(0));
        if (nchan_out) *nchan_out = nchan;
        if (channel < 0 || channel >= nchan) {
            std::cerr << "[read_ms_metadata] channel " << channel
                      << " out of range [0," << nchan << ") in '" << path << "'\n";
            return false;
        }
        const double freq = cf(IPosition(1, channel));
        const double wl   = C_LIGHT / freq;

        ArrayColumn<Double> uvwCol(ms, "UVW");
        Matrix<Double> uvw = uvwCol.getColumn();   // [3, total]
        double umax = 0.0;
        for (uInt i = 0; i < total; ++i) {
            const double b = std::sqrt(uvw(0, i) * uvw(0, i) + uvw(1, i) * uvw(1, i));
            if (b > umax) umax = b;
        }

        freq_out    = freq;
        umax_wl_out = umax / wl;                   // max |(u,v)| in wavelengths
        nrows_out   = static_cast<long>(total);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[read_ms_metadata] error in '" << path << "': " << e.what() << "\n";
        return false;
    }
}

} // namespace ddfacet

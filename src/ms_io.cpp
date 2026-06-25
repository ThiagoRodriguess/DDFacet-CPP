/**
 * @file ms_io.cpp
 * @brief Implementação da leitura de Measurement Sets via casacore.
 *
 * Dependência: casacore (já instalado em /usr/include/casacore).
 * Link: -lcasa_ms -lcasa_tables -lcasa_casa
 */
#include "ms_io.h"

#include <casacore/tables/Tables/Table.h>
#include <casacore/tables/Tables/ArrayColumn.h>
#include <casacore/casa/Arrays/Array.h>
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Cube.h>

#include <iostream>

namespace ddfacet {

bool read_ms(const std::string& path, VisibilitySet& vis, double& freq_out) {
    using namespace casacore;
    try {
        Table ms(path);
        const uInt nrow = ms.nrow();
        if (nrow == 0) {
            std::cerr << "[read_ms] MS vazio: " << path << "\n";
            return false;
        }

        // ── Frequência de referência (SPECTRAL_WINDOW.CHAN_FREQ[0]) ──────────
        Table spw(path + "/SPECTRAL_WINDOW");
        ArrayColumn<Double> chanFreq(spw, "CHAN_FREQ");
        Array<Double> cf = chanFreq.get(0);
        const double freq = cf(IPosition(1, 0));
        const double wl   = C_LIGHT / freq;        // comprimento de onda [m]

        // ── Colunas da tabela MAIN ──────────────────────────────────────────
        ArrayColumn<Double>  uvwCol(ms,    "UVW");
        ArrayColumn<Complex> dataCol(ms,   "DATA");
        ArrayColumn<Bool>    flagCol(ms,   "FLAG");
        ArrayColumn<Float>   weightCol(ms, "WEIGHT");

        // Leitura de colunas inteiras (mais eficiente que linha a linha)
        Matrix<Double>  uvw  = uvwCol.getColumn();     // [3, nrow]
        Cube<Complex>   data = dataCol.getColumn();    // [npol, nchan, nrow]
        Cube<Bool>      flag = flagCol.getColumn();    // [npol, nchan, nrow]
        Matrix<Float>   wgt  = weightCol.getColumn();  // [npol, nrow]

        vis.resize(nrow);
        for (uInt i = 0; i < nrow; ++i) {
            // UVW: metros → comprimentos de onda
            vis.u[i] = uvw(0, i) / wl;
            vis.v[i] = uvw(1, i) / wl;
            vis.w[i] = uvw(2, i) / wl;

            // DATA / FLAG: usa o 1º canal e 1ª polarização
            const Complex c = data(0, 0, i);
            vis.data[i]   = std::complex<float>(c.real(), c.imag());
            vis.flag[i]   = flag(0, 0, i);
            vis.weight[i] = wgt(0, i);
        }

        vis.freq_ref   = freq;
        vis.wavelength = wl;
        freq_out       = freq;

        std::cout << "  [read_ms] " << path << ": " << nrow
                  << " visibilidades lidas | freq " << freq / 1e9 << " GHz"
                  << " | wavelength " << wl << " m\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[read_ms] ERRO ao ler '" << path << "': " << e.what() << "\n";
        return false;
    }
}

} // namespace ddfacet

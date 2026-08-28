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
            std::cerr << "[read_ms] MS vazio: " << path << "\n";
            return false;
        }

        // Faixa de linhas [s, s+n) — distribuição MPI de um único MS.
        uInt s = (startrow < 0) ? 0u : static_cast<uInt>(startrow);
        if (s >= total) s = total - 1;
        uInt n = (nrow < 0) ? (total - s)
                            : std::min(static_cast<uInt>(nrow), total - s);

        // ── Frequência do CANAL pedido (SPECTRAL_WINDOW.CHAN_FREQ[channel]) ──
        // O canal é a unidade de trabalho independente (canal c → nó c no OMPC).
        // ATENÇÃO: cada canal tem a SUA frequência → o SEU comprimento de onda;
        // por isso a conversão UVW [m] → [λ] abaixo depende do canal escolhido.
        Table spw(path + "/SPECTRAL_WINDOW");
        ArrayColumn<Double> chanFreq(spw, "CHAN_FREQ");
        Array<Double> cf = chanFreq.get(0);
        const int nchan_spw = static_cast<int>(cf.shape()(0));
        if (channel < 0 || channel >= nchan_spw) {
            std::cerr << "[read_ms] canal " << channel << " fora da faixa [0,"
                      << nchan_spw << ") em '" << path << "'\n";
            return false;
        }
        const double freq = cf(IPosition(1, channel));
        const double wl   = C_LIGHT / freq;        // comprimento de onda [m]

        // ── Colunas da tabela MAIN (só a faixa de linhas deste rank) ─────────
        ArrayColumn<Double>  uvwCol(ms,    "UVW");
        ArrayColumn<Complex> dataCol(ms,   "DATA");
        ArrayColumn<Bool>    flagCol(ms,   "FLAG");
        ArrayColumn<Float>   weightCol(ms, "WEIGHT");

        const RefRows rows(s, s + n - 1);          // inclusivo → n linhas
        Matrix<Double>  uvw  = uvwCol.getColumnCells(rows);     // [3, n]
        Cube<Complex>   data = dataCol.getColumnCells(rows);    // [npol, nchan, n]
        Cube<Bool>      flag = flagCol.getColumnCells(rows);    // [npol, nchan, n]
        Matrix<Float>   wgt  = weightCol.getColumnCells(rows);  // [npol, n]

        // A coluna DATA pode ter menos canais que a SPECTRAL_WINDOW declara.
        const int nchan_data = static_cast<int>(data.shape()(1));
        if (channel >= nchan_data) {
            std::cerr << "[read_ms] canal " << channel << " ausente na coluna DATA ("
                      << nchan_data << " canais) em '" << path << "'\n";
            return false;
        }

        vis.resize(n);
        for (uInt i = 0; i < n; ++i) {
            // UVW: metros → comprimentos de onda
            vis.u[i] = uvw(0, i) / wl;
            vis.v[i] = uvw(1, i) / wl;
            vis.w[i] = uvw(2, i) / wl;

            // DATA / FLAG: polarização 0, canal `channel` (unidade distribuível)
            const Complex c = data(0, channel, i);
            vis.data[i]   = std::complex<float>(c.real(), c.imag());
            vis.flag[i]   = flag(0, channel, i);
            vis.weight[i] = wgt(0, i);
        }

        vis.freq_ref   = freq;
        vis.wavelength = wl;
        freq_out       = freq;

        std::cout << "  [read_ms] " << path << ": linhas [" << s << ".." << (s + n)
                  << ") = " << n << " visibilidades | canal " << channel << "/"
                  << nchan_spw << " | freq " << freq / 1e6 << " MHz\n";
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[read_ms] ERRO ao ler '" << path << "': " << e.what() << "\n";
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
            std::cerr << "[read_ms_metadata] canal " << channel << " fora da faixa [0,"
                      << nchan << ") em '" << path << "'\n";
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
        umax_wl_out = umax / wl;                    // max |(u,v)| em wavelengths
        nrows_out   = static_cast<long>(total);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[read_ms_metadata] ERRO em '" << path << "': " << e.what() << "\n";
        return false;
    }
}

} // namespace ddfacet

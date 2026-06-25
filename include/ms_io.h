/**
 * @file ms_io.h
 * @brief Leitura de Measurement Sets (formato CASA) via casacore.
 *
 * Isola a dependência do casacore: só este módulo (ms_io.cpp) inclui os headers
 * do casacore. O resto do pipeline continua sem essa dependência.
 */
#ifndef MS_IO_H
#define MS_IO_H

#include "ddfacet.h"
#include <string>

namespace ddfacet {

/**
 * @brief Lê um Measurement Set para um VisibilitySet.
 *
 * Lê as colunas UVW (metros), DATA, FLAG e WEIGHT da tabela MAIN e a
 * frequência de referência (CHAN_FREQ[0]) da subtabela SPECTRAL_WINDOW.
 * Converte UVW de metros para comprimentos de onda (÷ wavelength).
 *
 * @param path      Caminho do diretório .ms
 * @param vis       VisibilitySet de saída (preenchido)
 * @param freq_out  Frequência de referência [Hz]
 * @return true em sucesso, false em erro (mensagem em std::cerr)
 */
bool read_ms(const std::string& path, VisibilitySet& vis, double& freq_out);

} // namespace ddfacet

#endif // MS_IO_H

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
 * @brief Lê (uma faixa de linhas de) um Measurement Set para um VisibilitySet.
 *
 * Lê as colunas UVW (metros), DATA, FLAG e WEIGHT da tabela MAIN e a
 * frequência de referência (CHAN_FREQ[0]) da subtabela SPECTRAL_WINDOW.
 * Converte UVW de metros para comprimentos de onda (÷ wavelength).
 *
 * A faixa [startrow, startrow+nrow) permite DISTRIBUIR um único MS grande entre
 * ranks MPI: cada rank lê apenas a sua fatia de visibilidades.
 *
 * O CANAL é um parâmetro explícito: cada canal é uma unidade de trabalho
 * independente (frequência própria), o que prepara a distribuição futura
 * "canal c → nó c" no OpenMP Cluster. Note que a frequência usada é
 * CHAN_FREQ[channel] — trocar de canal muda o comprimento de onda e, portanto,
 * a conversão UVW [m] → [λ].
 *
 * @param path      Caminho do diretório .ms
 * @param vis       VisibilitySet de saída (preenchido)
 * @param freq_out  Frequência do canal lido [Hz]
 * @param startrow  Primeira linha a ler (padrão 0)
 * @param nrow      Número de linhas (padrão -1 = da startrow até o fim)
 * @param channel   Índice do canal espectral a ler (padrão 0)
 * @return true em sucesso, false em erro (mensagem em std::cerr)
 */
bool read_ms(const std::string& path, VisibilitySet& vis, double& freq_out,
             long startrow = 0, long nrow = -1, int channel = 0);

/**
 * @brief Lê apenas os metadados de um MS (sem carregar todas as visibilidades):
 * frequência, u_max em comprimentos de onda (para derivar o cell_size), e o
 * número total de linhas (para a distribuição MPI por faixa).
 *
 * @param path        Caminho do .ms
 * @param freq_out    Frequência do canal [Hz]
 * @param umax_wl_out max |(u,v)| em comprimentos de onda (no canal pedido)
 * @param nrows_out   número total de linhas do MS
 * @param channel     Índice do canal espectral (padrão 0)
 * @param nchan_out   (opcional) número total de canais do MS
 * @return true em sucesso
 */
bool read_ms_metadata(const std::string& path, double& freq_out,
                      double& umax_wl_out, long& nrows_out,
                      int channel = 0, int* nchan_out = nullptr);

} // namespace ddfacet

#endif // MS_IO_H

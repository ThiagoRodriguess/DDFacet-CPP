/**
 * =============================================================================
 * @file ddfacet.h
 * @brief DDFacet Sequential Imaging Pipeline - Implementação Educacional
 * 
 * Baseado no artigo: "SiPS IEEE 2022 - Monnier"
 * =============================================================================
 * 
 * ALGORITMO DDFACET SEQUENTIAL IMAGING PIPELINE
 * =============================================
 * 
 * Data: (x, v, I, J, K, PSF)
 *   - x   : Modelo do céu (sky model) - imagem que queremos reconstruir
 *   - v   : Visibilidades medidas pelo interferômetro
 *   - I   : Número de Facetas (divisões da imagem)
 *   - J   : Número de Measurement Sets (arquivos de dados)
 *   - K   : Número de Major Cycles
 *   - PSF : Point Spread Function (resposta do instrumento)
 * 
 * PSEUDOCÓDIGO:
 * -------------
 * Initialization(v);
 * 
 * for k in K MajorCycles do                    // LOOP 1: Major Cycles
 *     for j in J MS do                         // LOOP 2: Measurement Sets
 *         for i in I Facets do                 // LOOP 3: Predict (Facets)
 *             g_φi,MSj = FFT(x, i, j);
 *             v_φi,MSj = Degrid(g_φi,MSj, i, j);
 *         end
 *         δv_MSj = v_MSj - v̂_MSj;             // Residual de visibilidades
 *         for i in I Facets do                 // LOOP 4: Grid (Facets)
 *             g_φi,MSj = Grid(δv_MSj, i, j);
 *         end
 *     end
 *     for i in I Facets do                     // LOOP 5: Image (Facets)
 *         δy += FFT_inv(g_φi, i, j);
 *     end
 *     x = Deconvolution(δy, PSF);              // CLEAN
 * end
 * 
 * =============================================================================
 * EXPLICAÇÃO DETALHADA DOS LOOPS:
 * =============================================================================
 * 
 * LOOP 1 - MAJOR CYCLES (for k in K):
 * ------------------------------------
 * - É o loop mais externo do algoritmo
 * - Cada iteração refina progressivamente o modelo do céu
 * - Tipicamente K = 1 a 10 ciclos são suficientes
 * - Em cada major cycle:
 *   1. Prediz visibilidades a partir do modelo atual
 *   2. Calcula residuais (diferença entre medido e predito)
 *   3. Faz gridding dos residuais
 *   4. Cria imagem residual via FFT inversa
 *   5. Aplica deconvolução (CLEAN) para atualizar o modelo
 * 
 * LOOP 2 - MEASUREMENT SETS (for j in J):
 * ----------------------------------------
 * - Processa cada arquivo de dados (MS) separadamente
 * - Um MS contém visibilidades de uma observação
 * - Múltiplos MS podem vir de:
 *   - Diferentes épocas de observação
 *   - Diferentes bandas de frequência
 *   - Diferentes configurações do array
 * - Para cada MS, fazemos predict e grid das visibilidades
 * 
 * LOOP 3 - FACETS PREDICT (for i in I, primeiro):
 * ------------------------------------------------
 * - Divide a imagem em facetas para corrigir efeitos direcionais
 * - Para cada faceta i:
 *   1. FFT: Transforma o modelo da faceta para domínio UV (grade de Fourier)
 *   2. Degrid: Interpola a grade UV nas posições exatas das visibilidades
 * - Por que facetas? 
 *   - Campo de visão grande causa distorções (W-term)
 *   - Cada faceta é pequena o suficiente para aproximação 2D
 *   - Correções de fase são aplicadas por faceta
 * 
 * LOOP 4 - FACETS GRID (for i in I, segundo):
 * --------------------------------------------
 * - Após calcular δv (residual de visibilidades), faz gridding
 * - Grid: Coloca as visibilidades residuais na grade UV de cada faceta
 * - Operação inversa do Degrid
 * - Cada faceta acumula sua parte das visibilidades
 * 
 * LOOP 5 - FACETS IMAGE (for i in I, terceiro):
 * ----------------------------------------------
 * - Converte grades UV em imagem residual
 * - FFT_inv: Transforma cada grade UV para domínio de imagem
 * - δy acumula contribuições de todas as facetas
 * - Resultado: Imagem residual combinada de todas as facetas
 * 
 * DECONVOLUTION (CLEAN):
 * -----------------------
 * - Remove artefatos da PSF da imagem residual
 * - Encontra picos, adiciona ao modelo, subtrai PSF
 * - Atualiza x (modelo do céu) para o próximo major cycle
 * 
 * =============================================================================
 */

#ifndef DDFACET_H
#define DDFACET_H

#include <complex>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace ddfacet {

//=============================================================================
// CONSTANTES FÍSICAS
//=============================================================================

constexpr double C_LIGHT = 299792458.0;        // Velocidade da luz [m/s]
constexpr double PI = 3.14159265358979323846;
constexpr double DEG2RAD = PI / 180.0;
constexpr double RAD2DEG = 180.0 / PI;
constexpr double ARCSEC2RAD = DEG2RAD / 3600.0;

//=============================================================================
// PARÂMETROS DO KERNEL DE CONVOLUÇÃO (Grid / Degrid)
//=============================================================================
// Suporte (meio-largura) do kernel Gaussiano de (de)gridding, em pixels.
// O kernel cobre uma janela (2·W+1)×(2·W+1) ao redor do centro da visibilidade.
constexpr int    GRID_W_SUPPORT   = 5;
// Desvio-padrão do kernel Gaussiano, em pixels.
constexpr double GRID_KERNEL_SIGMA = 1.0;

//=============================================================================
// PARTE 1: ESTRUTURAS DE DADOS
//=============================================================================

/**
 * @brief Classe para representar uma imagem/grade 2D
 * 
 * Esta é a estrutura fundamental para armazenar:
 * - Imagens do céu (modelo x, residual δy)
 * - Grades UV (g_φi)
 * - PSF
 * 
 * Layout de memória: Row-major (linha por linha)
 * Acesso: array(i, j) onde i=coluna, j=linha
 */
template<typename T>
class Array2D {
public:
    int nx = 0;              // Largura (número de colunas)
    int ny = 0;              // Altura (número de linhas)
    std::vector<T> data;     // Dados armazenados linearmente
    
    // Construtores
    Array2D() = default;
    
    Array2D(int nx_, int ny_) 
        : nx(nx_), ny(ny_), data(nx_ * ny_, T(0)) {}
    
    Array2D(int nx_, int ny_, T init_val) 
        : nx(nx_), ny(ny_), data(nx_ * ny_, init_val) {}
    
    // Métodos de redimensionamento e inicialização
    void resize(int nx_, int ny_) {
        nx = nx_; 
        ny = ny_;
        data.resize(nx * ny);
    }
    
    void fill(T val) { 
        std::fill(data.begin(), data.end(), val); 
    }
    
    void zero() { 
        fill(T(0)); 
    }
    
    // Informações
    size_t size() const { return data.size(); }
    T* ptr() { return data.data(); }
    const T* ptr() const { return data.data(); }
    
    // Acesso aos elementos: (coluna, linha)
    T& operator()(int i, int j) { 
        return data[j * nx + i]; 
    }
    
    const T& operator()(int i, int j) const { 
        return data[j * nx + i]; 
    }
};

// Tipos comuns de arrays
using ImageF = Array2D<float>;                    // Imagem real (float)
using ImageD = Array2D<double>;                   // Imagem real (double)
using GridC = Array2D<std::complex<float>>;       // Grade UV complexa

//=============================================================================
// ESTRUTURA: Visibilidade
//=============================================================================
/**
 * @brief Representa uma única visibilidade medida
 * 
 * Uma visibilidade é a correlação entre duas antenas.
 * Ela mede uma componente de Fourier do céu.
 * 
 * Coordenadas (u, v, w):
 * - u, v: Projeção da baseline no plano perpendicular à fonte
 * - w: Componente na direção da fonte (causa o "w-term")
 * - Unidades: em comprimentos de onda (wavelengths)
 */
struct Visibility {
    double u;                    // Coordenada u [wavelengths]
    double v;                    // Coordenada v [wavelengths]
    double w;                    // Coordenada w [wavelengths]
    std::complex<float> data;    // Valor complexo da visibilidade
    float weight;                // Peso (qualidade/confiança)
    bool flag;                   // true = dado inválido/flagged
};

//=============================================================================
// ESTRUTURA: VisibilitySet (v no algoritmo)
//=============================================================================
/**
 * @brief Conjunto de todas as visibilidades de um Measurement Set
 * 
 * Corresponde a v_MSj no algoritmo.
 * Armazena todas as visibilidades medidas de um arquivo MS.
 */
struct VisibilitySet {
    std::vector<double> u;              // Coordenadas u de todas as visibilidades
    std::vector<double> v;              // Coordenadas v
    std::vector<double> w;              // Coordenadas w
    std::vector<std::complex<float>> data;  // Valores das visibilidades
    std::vector<float> weight;          // Pesos
    std::vector<bool> flag;             // Flags
    
    size_t nvis = 0;                    // Número total de visibilidades
    double freq_ref = 0.0;              // Frequência de referência [Hz]
    double wavelength = 0.0;            // Comprimento de onda [m]
    
    // Construtor
    VisibilitySet() = default;
    
    // Redimensiona para n visibilidades
    void resize(size_t n) {
        nvis = n;
        u.resize(n);
        v.resize(n);
        w.resize(n);
        data.resize(n);
        weight.resize(n);
        flag.resize(n, false);
    }
    
    // Número de visibilidades válidas (não flagged)
    size_t count_valid() const {
        size_t count = 0;
        for (size_t i = 0; i < nvis; ++i) {
            if (!flag[i]) count++;
        }
        return count;
    }
};

//=============================================================================
// ESTRUTURA: Facet (φi no algoritmo)
//=============================================================================
/**
 * @brief Representa uma faceta da imagem
 * 
 * A imagem completa é dividida em I facetas.
 * Cada faceta é uma região retangular do céu.
 * 
 * Por que usar facetas?
 * 1. Correção do W-term: Cada faceta é pequena, minimizando distorções
 * 2. Direction-Dependent Effects: Correções específicas por direção
 * 3. Paralelização: Facetas podem ser processadas independentemente
 * 
 * Coordenadas:
 * - (l, m): Coordenadas de direção no céu (cossenos diretores)
 * - (ra, dec): Posição central da faceta
 */
struct Facet {
    int id;                      // Identificador da faceta (0 a I-1)
    
    // Posição no grid de facetas
    int facet_i;                 // Índice na direção x
    int facet_j;                 // Índice na direção y
    
    // Centro da faceta (coordenadas celestes)
    double ra_center;            // Right Ascension do centro [rad]
    double dec_center;           // Declination do centro [rad]
    
    // Centro da faceta (cossenos diretores)
    double l_center;             // Cosseno diretor l
    double m_center;             // Cosseno diretor m

    // Efeito Dependente da Direção (DDE) — o "DD" de DDFacet.
    // Versão ESCALAR do formalismo RIME/Jones: um ganho complexo G por direção
    // (faceta), representando beam primário / ionosfera / erros de fase.
    // Aplicado como ×G no degrid e ×conj(G) no grid, o que PRESERVA a relação
    // adjunta entre os dois operadores. Padrão (1,0) = identidade (sem DDE).
    // Com múltiplas polarizações, G viraria uma matriz de Jones 2×2.
    std::complex<double> directional_gain;


    // Tamanho da faceta em pixels
    int npix_x;                  // Pixels na direção x
    int npix_y;                  // Pixels na direção y
    
    // Posição da faceta na imagem completa
    int offset_x;                // Offset x na imagem principal
    int offset_y;                // Offset y na imagem principal
    
    // Grade UV desta faceta (usada no PREDICT: imaging_fft escreve, degrid lê).
    GridC uv_grid;               // g_φi no algoritmo

    // Acumulador do residual gridado (passo GRID), SEPARADO de uv_grid — senão
    // o imaging_fft do próximo MS apagaria o que o grid acumulou. Lido pelo
    // imaging_fft_inv. Zerado no início de cada major cycle.
    GridC grid_acc;

    // Sub-imagem desta faceta
    ImageF image;                // Porção da imagem correspondente

    // Contribuição desta faceta às visibilidades preditas (v̂_φi).
    // Buffer PRÓPRIO por faceta → o degrid escreve aqui sem corrida, permitindo
    // paralelizar o laço de facetas; o main soma as contribuições em predicted.
    std::vector<std::complex<float>> pred_contrib;

    // Construtor padrão
    Facet() : id(0), facet_i(0), facet_j(0),
              ra_center(0), dec_center(0),
              l_center(0), m_center(0),
              directional_gain(1.0, 0.0),
              npix_x(0), npix_y(0),
              offset_x(0), offset_y(0) {}
};

//=============================================================================
// ESTRUTURA: MeasurementSet (MS_j no algoritmo)
//=============================================================================
/**
 * @brief Informações de um Measurement Set
 * 
 * Corresponde ao índice j no algoritmo (for j in J MS).
 * Contém metadados e visibilidades de uma observação.
 */
struct MeasurementSetInfo {
    int id;                          // Índice do MS (0 a J-1)
    std::string path;                // Caminho do arquivo MS
    VisibilitySet visibilities;      // Visibilidades deste MS (v_MSj)
    VisibilitySet predicted;         // Visibilidades preditas (v̂_MSj)
    VisibilitySet residual;          // Residual δv_MSj = v - v̂
    
    // Metadados
    double freq_min;                 // Frequência mínima [Hz]
    double freq_max;                 // Frequência máxima [Hz]
    double time_start;               // Tempo inicial [MJD]
    double time_end;                 // Tempo final [MJD]
    
    MeasurementSetInfo() : id(0), freq_min(0), freq_max(0), 
                           time_start(0), time_end(0) {}
};

//=============================================================================
// ESTRUTURA: DDFacetConfig - Parâmetros do Pipeline
//=============================================================================
/**
 * @brief Configuração do pipeline DDFacet
 * 
 * Contém todos os parâmetros necessários:
 * - K: Número de Major Cycles
 * - J: Número de Measurement Sets
 * - I: Número de Facetas (total = n_facets_x * n_facets_y)
 * - Parâmetros de imagem e deconvolução
 */
struct DDFacetConfig {
    // Parâmetros dos loops
    int n_major_cycles;          // K - número de major cycles
    int n_facets_x;              // Número de facetas na direção x
    int n_facets_y;              // Número de facetas na direção y
    
    // Parâmetros da imagem
    int image_size_x;            // Tamanho total da imagem (pixels) em x
    int image_size_y;            // Tamanho total da imagem (pixels) em y
    double cell_size_rad;        // Tamanho do pixel [rad]
    
    // Centro de fase (phase center)
    double ra0;                  // Right Ascension do centro [rad]
    double dec0;                 // Declination do centro [rad]
    
    // Parâmetros de deconvolução (CLEAN)
    int n_minor_iterations;      // Iterações por minor cycle
    double clean_gain;           // γ - loop gain (tipicamente 0.1)
    double clean_threshold;      // Threshold para parar
    
    // Construtor com valores padrão
    DDFacetConfig() 
        : n_major_cycles(5)
        , n_facets_x(1)
        , n_facets_y(1)
        , image_size_x(512)
        , image_size_y(512)
        , cell_size_rad(ARCSEC2RAD)  // 1 arcsec por padrão
        , ra0(0.0)
        , dec0(0.0)
        , n_minor_iterations(1000)
        , clean_gain(0.1)
        , clean_threshold(0.0)
    {}
    
    // Número total de facetas (I no algoritmo)
    int total_facets() const {
        return n_facets_x * n_facets_y;
    }
    
    // Tamanho de cada faceta em pixels
    int facet_size_x() const {
        return image_size_x / n_facets_x;
    }
    
    int facet_size_y() const {
        return image_size_y / n_facets_y;
    }
};

//=============================================================================
// ESTRUTURA: DDFacetState - Estado do Pipeline
//=============================================================================
/**
 * @brief Estado atual do pipeline DDFacet
 * 
 * Contém todas as variáveis do algoritmo:
 * - x: Modelo do céu (sky model)
 * - δy: Imagem residual
 * - PSF: Point Spread Function
 * - Facetas e Measurement Sets
 */
struct DDFacetState {
    // Configuração
    DDFacetConfig config;
    
    // Imagens principais
    ImageF x;                    // Modelo do céu (sky model)
    ImageF delta_y;              // Imagem residual (δy)
    ImageF psf;                  // Point Spread Function
    
    // Facetas (I facetas no total)
    std::vector<Facet> facets;
    
    // Measurement Sets (J no total)
    std::vector<MeasurementSetInfo> measurement_sets;
    
    // Estado da iteração
    int current_major_cycle;     // k atual
    int current_ms;              // j atual
    int current_facet;           // i atual
    
    // Estatísticas
    double peak_residual;        // Pico do residual atual
    int total_components;        // Componentes CLEAN encontrados
    
    DDFacetState() 
        : current_major_cycle(0)
        , current_ms(0)
        , current_facet(0)
        , peak_residual(0.0)
        , total_components(0)
    {}
};

//=============================================================================
// PARTE 2: FUNÇÃO INITIALIZATION(v)
//=============================================================================
/**
 * @brief Inicializa o pipeline DDFacet
 * 
 * Esta função corresponde a "Initialization(v)" no algoritmo.
 * 
 * O que ela faz:
 * 1. Inicializa o modelo do céu x com zeros
 * 2. Cria e configura as facetas
 * 3. Aloca memória para grades UV
 * 4. Prepara estruturas para as visibilidades
 * 5. Cria/carrega a PSF
 * 
 * @param state Estado do pipeline (será inicializado)
 * @param config Configuração dos parâmetros
 * 
 * Após esta função, o pipeline está pronto para o primeiro major cycle.
 */
void initialization(DDFacetState& state, const DDFacetConfig& config);

/**
 * @brief Cria e configura as facetas
 * 
 * Divide a imagem em n_facets_x × n_facets_y facetas.
 * Cada faceta recebe:
 * - Sua posição na imagem
 * - Seu centro (ra, dec) e (l, m)
 * - Sua grade UV alocada
 * 
 * @param state Estado do pipeline
 */
void create_facets(DDFacetState& state);

/**
 * @brief Cria uma PSF gaussiana simples para testes
 * 
 * @param size Tamanho da PSF (size × size)
 * @param sigma Desvio padrão da gaussiana
 * @return PSF normalizada (pico = 1.0)
 */
ImageF create_gaussian_psf(int size, double sigma);

/**
 * @brief Imprime informações do estado do pipeline
 * 
 * @param state Estado do pipeline
 */
void print_state_info(const DDFacetState& state);


//=============================================================================
// PARTE 3: FFT UTILITIES — Cooley-Tukey Radix-2, pure C++17
//=============================================================================

/**
 * @brief Retorna a menor potência de 2 maior ou igual a n.
 *
 * Exemplo:  next_pow2(5) = 8,  next_pow2(8) = 8
 */
size_t next_pow2(size_t n);

/**
 * @brief FFT / IFFT 1D in-place — Cooley-Tukey iterativo (radix-2).
 *
 * @param data    Vetor de complexos. DEVE ter tamanho potência de 2.
 * @param inverse false → FFT forward,  true → IFFT (normalizada por 1/N).
 *
 * Complexidade: O(N log N)
 * Precisão:     double (internamente)
 */
void fft_1d(std::vector<std::complex<double>>& data, bool inverse);

/**
 * @brief FFT / IFFT 2D separável sobre um GridC (float complexo).
 *
 * Realiza FFT linha a linha e depois coluna a coluna (algoritmo separável).
 * O tamanho nx e ny DEVEM ser potências de 2; caso contrário, a função
 * faz zero-padding automaticamente e retorna com o tamanho original.
 *
 * @param grid    Grade UV (nx × ny complexo float). Modificada in-place.
 * @param forward true → FFT forward (imagem → UV),
 *                false → IFFT (UV → imagem), com normalização 1/(nx*ny).
 */
void fft_2d(GridC& grid, bool forward);

/**
 * @brief fftshift 2D: troca os 4 quadrantes para colocar DC no centro.
 *
 * Equivalente ao numpy.fft.fftshift().
 * Após FFT, a frequência zero fica em (0,0); após fftshift fica em (nx/2, ny/2).
 * Necessário porque o DDFacet trabalha com a grade UV centrada no DC.
 *
 *   +-------+-------+          +-------+-------+
 *   | Q3    | Q4    |          | Q1    | Q2    |
 *   +-------+-------+  ─────►  +-------+-------+
 *   | Q1    | Q2    |          | Q3    | Q4    |
 *   +-------+-------+          +-------+-------+
 */
void fftshift_2d(GridC& grid);

/**
 * @brief ifftshift 2D: operação inversa de fftshift.
 *
 * Deve ser aplicado ANTES do IFFT para desfazer o fftshift.
 * Para tamanhos pares, fftshift e ifftshift são equivalentes.
 */
void ifftshift_2d(GridC& grid);

//=============================================================================
// PARTE 4: PASSO FFT DO ALGORITMO 1
//   Linha do pseudocódigo:  ĝφi,MSj = FFT(x̂, i, j)
//=============================================================================

/**
 * @brief Implementa:  ĝφi,MSj = FFT(x̂, i, j)
 *
 * Este passo corresponde à PREDIÇÃO no domínio UV (Predict step).
 * Dado o sky model atual x̂, gera as visibilidades previstas na forma
 * de uma grade UV para a faceta i.
 *
 * Passos implementados:
 *   1. Extrai sub-imagem da faceta i do sky model x̂.
 *   2. Aplica rotação de fase do centro da faceta (l0, m0):
 *        pixel(ix, iy) *= exp(-2πi * (l0 * u_norm_x + m0 * v_norm_y))
 *      Isso desloca a origem do UV-grid para o centro da faceta.
 *   3. Converte float → complex<float> (imagem real → grade complexa).
 *   4. Aplica FFT 2D forward (Cooley-Tukey).
 *   5. Aplica fftshift (DC no centro, convenção DDFacet).
 *   6. Armazena resultado em state.facets[facet_idx].uv_grid.
 *
 * TODO (Fase 2):
 *   - Aplicar W-term por pixel: exp(-2πi * w * (sqrt(1-l²-m²) - 1))
 *     necessário quando houver dados reais com baselines não-coplanares.
 *
 * @param state      Estado completo do pipeline.
 * @param facet_idx  Índice da faceta (i no loop do algoritmo).
 * @param ms_idx     Índice do Measurement Set (j no loop, usado para log).
 */
void imaging_fft(DDFacetState& state, int facet_idx, int ms_idx);

//=============================================================================
// PARTE 5: PASSO FFT_inv DO ALGORITMO 1
//   Linha do pseudocódigo:  δy += FFT_inv(gφi, i, j)
//=============================================================================

/**
 * @brief Implementa:  δy += FFT_inv(gφi, i, j)
 *
 * Este passo converte a grade UV acumulada da faceta i de volta ao domínio
 * de imagem e acumula na imagem residual δy.
 *
 * Passos implementados:
 *   1. Lê state.facets[facet_idx].uv_grid (grade UV acumulada pelo Grid step).
 *   2. Aplica ifftshift (desfaz o fftshift anterior).
 *   3. Aplica IFFT 2D (Cooley-Tukey inverso, normalizado por 1/N).
 *   4. Remove rotação de fase do centro da faceta (derotação):
 *        pixel(ix, iy) *= exp(+2πi * (l0 * u_norm_x + m0 * v_norm_y))
 *   5. Acumula a parte REAL do resultado em state.delta_y na posição
 *      correta (offset_x, offset_y) da faceta dentro da imagem completa.
 *
 * TODO (Fase 2):
 *   - Remover W-term por pixel antes de acumular.
 *
 * @param state      Estado completo do pipeline.
 * @param facet_idx  Índice da faceta (i no loop do algoritmo).
 */
void imaging_fft_inv(DDFacetState& state, int facet_idx);

//=============================================================================
// PARTE 6: PASSO DEGRID DO ALGORITMO 1
//   Linha do pseudocódigo:  v̂φi,MSj = Degrid(ĝφi,MSj, i, j)
//=============================================================================

/**
 * @brief Implementa:  v̂φi,MSj = Degrid(ĝφi,MSj, i, j)
 *
 * Operação de "amostragem" (interpolação): converte a grade UV UNIFORME da
 * faceta (state.facets[facet_idx].uv_grid, já em convenção fftshift com o DC
 * no centro) nas visibilidades PREDITAS nas posições NÃO-uniformes (u,v) de
 * cada visibilidade do Measurement Set j.
 *
 * Para cada visibilidade k não-flagged:
 *   1. Converte (u_k, v_k) [wavelengths] → posição em pixels na grade:
 *        ix_c = u_k · nx · cell_size_rad + nx/2
 *        iy_c = v_k · ny · cell_size_rad + ny/2
 *   2. Convolui com o kernel Gaussiano no suporte W:
 *        v̂_k = Σ uv_grid(ix, iy) · C(du, dv)   /   Σ C(du, dv)
 *      onde C(du,dv) = exp(-(du²+dv²)/(2σ²)),  du = ix - ix_c.
 *   3. ACUMULA em state.measurement_sets[ms_idx].predicted.data[k]
 *      (acumulação entre facetas: a visibilidade total do céu é a soma das
 *       contribuições de todas as facetas — por isso `predicted` deve ser
 *       zerada ANTES do laço de facetas do passo Predict).
 *
 * Índices fora da grade sofrem wrapping (módulo nx/ny), refletindo a
 * periodicidade da DFT.
 *
 * @param state      Estado completo do pipeline.
 * @param facet_idx  Índice da faceta (i no laço do algoritmo).
 * @param ms_idx     Índice do Measurement Set (j no laço).
 */
void degrid(DDFacetState& state, int facet_idx, int ms_idx);

//=============================================================================
// PARTE 7: PASSO GRID DO ALGORITMO 1
//   Linha do pseudocódigo:  gφi,MSj = Grid(δvMSj, i, j)
//=============================================================================

/**
 * @brief Implementa:  gφi,MSj = Grid(δvMSj, i, j)
 *
 * Operação ADJUNTA do Degrid ("anterpolação"): distribui cada visibilidade
 * residual δv_k (state.measurement_sets[ms_idx].residual.data[k]) na grade UV
 * da faceta, ponderada pelo kernel de convolução (conjugado).
 *
 * Para cada visibilidade k não-flagged:
 *   1. Converte (u_k, v_k) → posição em pixels (ix_c, iy_c) (mesma fórmula do
 *      Degrid).
 *   2. ACUMULA no suporte W:
 *        uv_grid(ix, iy) += δv_k · conj(C(du, dv))
 *
 * IMPORTANTE: acumula em facet.uv_grid (múltiplos MS se somam). A grade deve
 * ser zerada no início de cada major cycle, antes de qualquer chamada a grid().
 *
 * @param state      Estado completo do pipeline.
 * @param facet_idx  Índice da faceta (i no laço do algoritmo).
 * @param ms_idx     Índice do Measurement Set (j no laço).
 */
void grid(DDFacetState& state, int facet_idx, int ms_idx);

//=============================================================================
// PARTE 8: RESIDUAL DE VISIBILIDADES
//   Linha do pseudocódigo:  δvMSj = vMSj − v̂MSj
//=============================================================================

/**
 * @brief Calcula o residual de visibilidades δv = v − v̂.
 *
 * Convenção (Algoritmo 1, linha 30): δv = v_medido − v̂_predito.
 * Esta é a convenção que faz o major cycle CONVERGIR: quando o modelo x̂
 * cresce em direção à fonte verdadeira, v̂ → v e o residual → 0.
 * (A "imagem suja" do residual fica com sinal positivo para fontes positivas,
 *  permitindo que o CLEAN adicione componentes positivas ao modelo.)
 *
 * Copia também (u,v,w,flag) de `visibilities` para `residual` e zera o
 * residual das visibilidades flagged.
 *
 * @param state   Estado completo do pipeline.
 * @param ms_idx  Índice do Measurement Set (j).
 */
void compute_residual(DDFacetState& state, int ms_idx);

//=============================================================================
// PARTE 9: DECONVOLUTION (Högbom CLEAN)
//   Linha do pseudocódigo:  x̂ = Deconvolution(δy, PSF)
//=============================================================================

/**
 * @brief Implementa:  x̂ = Deconvolution(δy, PSF) — Högbom CLEAN.
 *
 * A cada iteração (minor cycle):
 *   1. Encontra o pico de |δy| em (peak_x, peak_y).
 *   2. Adiciona γ·δy(peak) ao modelo x̂.
 *   3. Subtrai a PSF escalonada de δy, centrada no pico.
 *   4. Repete até n_minor_iterations ou |pico| < clean_threshold.
 *
 * Usa state.psf (pico normalizado em 1.0 no centro (nx/2, ny/2)) e os
 * parâmetros state.config.{clean_gain, n_minor_iterations, clean_threshold}.
 * Atualiza state.total_components e state.peak_residual.
 *
 * @param state  Estado completo do pipeline.
 */
void deconvolution(DDFacetState& state);

} // namespace ddfacet

#endif // DDFACET_H

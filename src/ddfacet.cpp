/**
 * =============================================================================
 * @file ddfacet.cpp
 * @brief Implementação das funções do DDFacet Sequential Imaging Pipeline
 * =============================================================================
 */

#include "ddfacet.h"
#include <cstdlib>
#include <cmath>
#include <iomanip>

namespace ddfacet {

/**
 * @brief Liga/desliga o TERMO-W (interruptor de demonstração: DDF_NOW=1 desliga).
 *
 * Existe para poder mostrar lado a lado o efeito do termo w·(n0-1) na fase da
 * faceta. Por padrão o termo está ATIVO; DDF_NOW=1 volta ao comportamento antigo,
 * em que o w era lido do MS e ignorado no cálculo.
 */
static bool ddf_use_w_term() {
    static const bool on = (std::getenv("DDF_NOW") == nullptr);
    return on;
}

//=============================================================================
// IMPLEMENTAÇÃO: Initialization(v)
//=============================================================================

void initialization(DDFacetState& state, const DDFacetConfig& config) {
    /**
     * INITIALIZATION(v) - Passo inicial do algoritmo
     * 
     * Esta função prepara todo o estado necessário antes de começar
     * os major cycles. É chamada apenas UMA VEZ no início.
     * 
     * Passos:
     * 1. Armazena a configuração
     * 2. Inicializa x (modelo do céu) com zeros
     * 3. Inicializa δy (residual) com zeros
     * 4. Cria a PSF
     * 5. Cria as facetas
     * 6. Reseta contadores
     */
    
    std::cout << "========================================\n";
    std::cout << "  INITIALIZATION(v)\n";
    std::cout << "========================================\n\n";
    
    // Passo 1: Armazena configuração
    state.config = config;
    
    std::cout << "Configuração:\n";
    std::cout << "  - Major Cycles (K): " << config.n_major_cycles << "\n";
    std::cout << "  - Facetas: " << config.n_facets_x << " x " << config.n_facets_y;
    std::cout << " = " << config.total_facets() << " total\n";
    std::cout << "  - Tamanho da imagem: " << config.image_size_x << " x " << config.image_size_y << " pixels\n";
    std::cout << "  - Tamanho da faceta: " << config.facet_size_x() << " x " << config.facet_size_y() << " pixels\n";
    std::cout << "  - Tamanho do pixel: " << config.cell_size_rad / ARCSEC2RAD << " arcsec\n";
    std::cout << "  - CLEAN gain (γ): " << config.clean_gain << "\n";
    std::cout << "  - Minor iterations: " << config.n_minor_iterations << "\n\n";
    
    // Passo 2: Inicializa x (modelo do céu) com zeros
    // x começa vazio - será preenchido pela deconvolução
    std::cout << "Inicializando modelo do céu (x)...\n";
    state.x.resize(config.image_size_x, config.image_size_y);
    state.x.zero();
    std::cout << "  - x inicializado: " << state.x.nx << " x " << state.x.ny << " pixels\n";
    
    // Passo 3: Inicializa δy (imagem residual) com zeros
    // δy será preenchido após FFT inversa das grades UV
    std::cout << "Inicializando imagem residual (δy)...\n";
    state.delta_y.resize(config.image_size_x, config.image_size_y);
    state.delta_y.zero();
    std::cout << "  - δy inicializado: " << state.delta_y.nx << " x " << state.delta_y.ny << " pixels\n";
    
    // Passo 4: Cria a PSF
    // Por enquanto, uma PSF gaussiana simples para testes
    std::cout << "Criando PSF...\n";
    int psf_size = std::min(config.image_size_x, config.image_size_y);
    double psf_sigma = 3.0;  // Sigma em pixels
    state.psf = create_gaussian_psf(psf_size, psf_sigma);
    std::cout << "  - PSF criada: " << state.psf.nx << " x " << state.psf.ny << " pixels\n";
    std::cout << "  - PSF sigma: " << psf_sigma << " pixels\n";
    
    // Passo 5: Cria as facetas
    std::cout << "\nCriando facetas...\n";
    create_facets(state);
    
    // Passo 6: Reseta contadores
    state.current_major_cycle = 0;
    state.current_ms = 0;
    state.current_facet = 0;
    state.peak_residual = 0.0;
    state.total_components = 0;
    
    std::cout << "\n========================================\n";
    std::cout << "  INITIALIZATION COMPLETA\n";
    std::cout << "========================================\n\n";
}

//=============================================================================
// IMPLEMENTAÇÃO: create_facets
//=============================================================================

void create_facets(DDFacetState& state) {
    /**
     * CREATE_FACETS - Cria e configura as I facetas
     * 
     * A imagem é dividida em uma grade de facetas:
     * 
     *   +-------+-------+-------+
     *   | (0,2) | (1,2) | (2,2) |
     *   +-------+-------+-------+
     *   | (0,1) | (1,1) | (2,1) |
     *   +-------+-------+-------+
     *   | (0,0) | (1,0) | (2,0) |
     *   +-------+-------+-------+
     * 
     * Cada faceta (facet_i, facet_j) tem:
     * - Posição na imagem completa (offset_x, offset_y)
     * - Centro em coordenadas celestes (ra, dec) e (l, m)
     * - Sua própria grade UV para gridding/degridding
     */
    
    const DDFacetConfig& cfg = state.config;
    
    int n_facets = cfg.total_facets();
    state.facets.resize(n_facets);
    
    int facet_nx = cfg.facet_size_x();
    int facet_ny = cfg.facet_size_y();
    
    int id = 0;
    for (int fj = 0; fj < cfg.n_facets_y; ++fj) {
        for (int fi = 0; fi < cfg.n_facets_x; ++fi) {
            Facet& facet = state.facets[id];
            
            // Identificação
            facet.id = id;
            facet.facet_i = fi;
            facet.facet_j = fj;
            
            // Tamanho em pixels
            facet.npix_x = facet_nx;
            facet.npix_y = facet_ny;
            
            // Posição na imagem completa
            facet.offset_x = fi * facet_nx;
            facet.offset_y = fj * facet_ny;
            
            // Centro da faceta em pixels (relativo à imagem completa)
            double center_pix_x = facet.offset_x + facet_nx / 2.0;
            double center_pix_y = facet.offset_y + facet_ny / 2.0;
            
            // Centro da imagem completa em pixels
            double img_center_x = cfg.image_size_x / 2.0;
            double img_center_y = cfg.image_size_y / 2.0;
            
            // Deslocamento do centro da faceta em relação ao centro da imagem
            double delta_pix_x = center_pix_x - img_center_x;
            double delta_pix_y = center_pix_y - img_center_y;
            
            // Centro da faceta em cossenos diretores (l₀, m₀), na MESMA convenção
            // da DFT de medida: l = (px − N/2)·cell. Usado como deslocamento de
            // FASE da faceta no degrid/grid: exp(∓2πi(u·l₀ + v·m₀)).
            facet.l_center = delta_pix_x * cfg.cell_size_rad;
            facet.m_center = delta_pix_y * cfg.cell_size_rad;
            
            // Converte para (ra, dec) aproximado (válido para campo pequeno)
            facet.ra_center = cfg.ra0 + facet.l_center / std::cos(cfg.dec0);
            facet.dec_center = cfg.dec0 + facet.m_center;
            
            // Aloca grade UV para esta faceta (predict) e o acumulador do grid
            facet.uv_grid.resize(facet_nx, facet_ny);
            facet.uv_grid.zero();
            facet.grid_acc.resize(facet_nx, facet_ny);
            facet.grid_acc.zero();
            
            // Aloca sub-imagem
            facet.image.resize(facet_nx, facet_ny);
            facet.image.zero();
            
            std::cout << "  Faceta " << id << " (" << fi << "," << fj << "):\n";
            std::cout << "    - Offset: (" << facet.offset_x << ", " << facet.offset_y << ")\n";
            std::cout << "    - Tamanho: " << facet.npix_x << " x " << facet.npix_y << " pixels\n";
            std::cout << "    - Centro (l,m): (" << facet.l_center/ARCSEC2RAD << "\", " 
                      << facet.m_center/ARCSEC2RAD << "\")\n";
            
            id++;
        }
    }
    
    std::cout << "\n  Total de facetas criadas: " << n_facets << "\n";
}

//=============================================================================
// IMPLEMENTAÇÃO: create_gaussian_psf
//=============================================================================

ImageF create_gaussian_psf(int size, double sigma) {
    /**
     * CREATE_GAUSSIAN_PSF - Cria uma PSF gaussiana 2D
     * 
     * A PSF (Point Spread Function) descreve como uma fonte pontual
     * aparece na imagem devido à resposta do instrumento.
     * 
     * Para um interferômetro real, a PSF (ou "dirty beam") é determinada
     * pela cobertura UV. Aqui usamos uma gaussiana para testes.
     * 
     * Fórmula: PSF(x,y) = exp(-((x-cx)² + (y-cy)²) / (2σ²))
     * 
     * A PSF é normalizada para ter pico = 1.0 no centro.
     */
    
    ImageF psf(size, size);
    
    double cx = size / 2.0;  // Centro x
    double cy = size / 2.0;  // Centro y
    double sigma2 = 2.0 * sigma * sigma;
    
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < size; ++i) {
            double dx = i - cx;
            double dy = j - cy;
            double r2 = dx * dx + dy * dy;
            psf(i, j) = static_cast<float>(std::exp(-r2 / sigma2));
        }
    }
    
    // Normaliza para pico = 1.0
    float peak = psf(size/2, size/2);
    if (peak > 0) {
        for (size_t idx = 0; idx < psf.size(); ++idx) {
            psf.data[idx] /= peak;
        }
    }
    
    return psf;
}

//=============================================================================
// IMPLEMENTAÇÃO: print_state_info
//=============================================================================

void print_state_info(const DDFacetState& state) {
    /**
     * PRINT_STATE_INFO - Imprime informações do estado atual
     * 
     * Útil para debugging e acompanhamento do progresso.
     */
    
    std::cout << "\n========================================\n";
    std::cout << "  ESTADO DO PIPELINE\n";
    std::cout << "========================================\n";
    
    std::cout << "\nConfiguração:\n";
    std::cout << "  - Major Cycles (K): " << state.config.n_major_cycles << "\n";
    std::cout << "  - Facetas (I): " << state.config.total_facets() << "\n";
    std::cout << "  - Measurement Sets (J): " << state.measurement_sets.size() << "\n";
    
    std::cout << "\nProgresso:\n";
    std::cout << "  - Major Cycle atual: " << state.current_major_cycle << "/" << state.config.n_major_cycles << "\n";
    std::cout << "  - MS atual: " << state.current_ms << "\n";
    std::cout << "  - Faceta atual: " << state.current_facet << "\n";
    
    std::cout << "\nEstatísticas:\n";
    std::cout << "  - Pico do residual: " << state.peak_residual << "\n";
    std::cout << "  - Componentes CLEAN: " << state.total_components << "\n";
    
    // Estatísticas do modelo
    if (state.x.size() > 0) {
        float x_min = state.x.data[0], x_max = state.x.data[0];
        double x_sum = 0;
        for (size_t i = 0; i < state.x.size(); ++i) {
            x_min = std::min(x_min, state.x.data[i]);
            x_max = std::max(x_max, state.x.data[i]);
            x_sum += state.x.data[i];
        }
        std::cout << "\nModelo (x):\n";
        std::cout << "  - Min: " << x_min << "\n";
        std::cout << "  - Max: " << x_max << "\n";
        std::cout << "  - Soma: " << x_sum << "\n";
    }
    
    std::cout << "\n========================================\n\n";
}


//=============================================================================
// IMPLEMENTAÇÃO: imaging_fft  —  ĝφi,MSj = FFT(x̂, i, j)
//=============================================================================

void imaging_fft(DDFacetState& state, int facet_idx, int ms_idx) {
    /**
     * CONTEXTO NO ALGORITMO 1:
     * ─────────────────────────
     * for k in K_MajorCycles do
     *   for j in J_MS do
     *     for i in I_Facets do          ← estamos aqui
     *       ĝφi,MSj = FFT(x̂, i, j);   ← esta função
     *       v̂φi,MSj = Degrid(...)      ← próximo passo (Fase 1, Passo 2)
     *
     * O QUE ESTA FUNÇÃO FAZ:
     * ──────────────────────
     * Transforma a sub-imagem da faceta i do sky model (x̂) para o domínio UV,
     * gerando a grade ĝφi,MSj. O Degrid irá então amostrar essa grade nas
     * posições (u,v) reais das visibilidades do MS j.
     *
     * FASE DO CENTRO (Phase Rotation):
     * ──────────────────────────────────
     * O DDFacet usa um grid separado por faceta. Cada faceta é centrada em
     * (l₀, m₀) em vez de (0,0). Para isso, antes da FFT aplica-se uma
     * rotação de fase equivalente a deslocar a origem do grid UV:
     *
     *   pixel(ix, iy) ← pixel(ix, iy) · exp(-2πi · (l₀·uₓ + m₀·vᵧ))
     *
     * onde (uₓ, vᵧ) são as coordenadas UV normalizadas correspondentes ao
     * pixel (ix, iy) no espaço de imagem:
     *   uₓ = (ix - cx) / nx,   vᵧ = (iy - cy) / ny
     *
     * Isso é equivalente a shiftar a grade UV pelo vetor (l₀, m₀).
     *
     * Nota: para l₀ = m₀ = 0 (faceta central), a rotação é identidade.
     */

    Facet& facet = state.facets[static_cast<size_t>(facet_idx)];
    const DDFacetConfig& cfg = state.config;

    const int nx = facet.npix_x;
    const int ny = facet.npix_y;

    // ── Passo 1: Copia sub-imagem da faceta i a partir de state.x ─────────────
    // state.x contém o sky model completo. A faceta i ocupa a região:
    //   [offset_x .. offset_x+nx-1] × [offset_y .. offset_y+ny-1]
    //
    // Inicializa a grade UV da faceta com zeros
    facet.uv_grid.resize(nx, ny);
    facet.uv_grid.zero();

    // PARALELIZAÇÃO (OpenMP): cada pixel (ix,iy) é independente — escreve a sua
    // própria célula da grade UV. collapse(2) distribui o produto dos dois laços.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            // Posição no sky model completo
            int gx = facet.offset_x + ix;
            int gy = facet.offset_y + iy;

            float pixel_val = 0.0f;
            // Boundary check (por precaução)
            if (gx >= 0 && gx < cfg.image_size_x &&
                gy >= 0 && gy < cfg.image_size_y) {
                pixel_val = state.x(gx, gy);
            }

            // A sub-imagem da faceta vai direto para a grade (real → complexo).
            // O deslocamento de fase da faceta NÃO é aplicado aqui como rampa no
            // espaço-imagem (isso só shiftava a grade em pixels). Ele é aplicado
            // no degrid/grid em função do (u,v) REAL de cada visibilidade —
            // exp(∓2πi(u·l₀+v·m₀)) — que é a forma correta de faceamento.
            facet.uv_grid(ix, iy) = std::complex<float>(pixel_val, 0.0f);
        }
    }

    // ── Passo 2b: ifftshift do domínio-imagem (centragem da fase) ─────────────
    // Coloca o centro da faceta (cx, cy) na origem do FFT (pixel 0). Sem isto
    // a referência de fase ficaria no canto (pixel 0) e a apodização do kernel
    // de gridding atenuaria fortemente fontes longe do canto. Com a centragem,
    // o centro da imagem é o phase center e a correção de grid fica simétrica.
    ifftshift_2d(facet.uv_grid);

    // ── Passo 3: FFT 2D forward ───────────────────────────────────────────────
    // Transforma grid(nx × ny) de domínio imagem → domínio UV (Fourier)
    // Resultado: facet.uv_grid[u, v] = DFT2D{ sub-imagem com phase rotation }
    fft_2d(facet.uv_grid, /*forward=*/true);

    // ── Passo 4: fftshift ─────────────────────────────────────────────────────
    // Após FFT, DC está em (0,0). Aplica fftshift para mover DC ao centro.
    // Convenção DDFacet: grade UV centrada em DC.
    fftshift_2d(facet.uv_grid);

    // Log
    std::cout << "    [imaging_fft] Faceta " << facet_idx
              << " (MS " << ms_idx << "): "
              << nx << "×" << ny << " px"
              << " | l₀=" << std::fixed << std::setprecision(2)
              << facet.l_center / ARCSEC2RAD << "\" "
              << " m₀=" << facet.m_center / ARCSEC2RAD << "\""
              << " → grade UV gerada.\n";
}

//=============================================================================
// IMPLEMENTAÇÃO: imaging_fft_inv  —  δy += FFT_inv(gφi, i, j)
//=============================================================================

void imaging_fft_inv(DDFacetState& state, int facet_idx) {
    /**
     * CONTEXTO NO ALGORITMO 1:
     * ─────────────────────────
     * for k in K_MajorCycles do
     *   ...
     *   for i in I_Facets do           ← estamos aqui
     *     δy += FFT_inv(gφi, i, j);   ← esta função
     *   end
     *   x̂ = Deconvolution(δy, PSF)
     *
     * O QUE ESTA FUNÇÃO FAZ:
     * ──────────────────────
     * Após o Grid step acumular visibilidades residuais na grade UV da faceta,
     * esta função converte a grade UV de volta para o domínio de imagem
     * e acumula o resultado na imagem residual δy.
     *
     * ACUMULAÇÃO ("+=" no pseudocódigo):
     * ────────────────────────────────────
     * δy é zerada NO INÍCIO de cada major cycle (antes de processar todos os MS).
     * Cada faceta contribui com sua parcela. Ao final do loop de facetas,
     * δy contém a imagem residual completa usada pelo CLEAN.
     *
     * DEROTAÇÃO DE FASE:
     * ───────────────────
     * Operação inversa da phase rotation aplicada no imaging_fft.
     * Após IFFT, multiplica cada pixel por exp(+2πi · (l₀·uₓ + m₀·vᵧ)):
     *   pixel_derotated(ix, iy) = pixel_ifft(ix, iy) · exp(+2πi · (...))
     *
     * Apenas a parte REAL é acumulada em δy (a imagem do céu é real).
     */

    Facet& facet = state.facets[static_cast<size_t>(facet_idx)];
    const DDFacetConfig& cfg = state.config;

    const int nx = facet.npix_x;
    const int ny = facet.npix_y;

    // Lê o ACUMULADOR do grid (residual gridado de todos os MS), não o uv_grid
    // (que é sobrescrito pelo predict). Cópia local para não destruir o acumulador.
    GridC work_grid = facet.grid_acc;

    // ── Passo 1: ifftshift ────────────────────────────────────────────────────
    // Desfaz o fftshift aplicado após o Grid step.
    // Necessário para que o DC fique em (0,0) antes da IFFT.
    ifftshift_2d(work_grid);

    // ── Passo 2: IFFT 2D ──────────────────────────────────────────────────────
    // Converte domínio UV (Fourier) → domínio imagem.
    // forward=false → IFFT, com normalização 1/(nx*ny).
    fft_2d(work_grid, /*forward=*/false);

    // ── Passo 2b: fftshift do domínio-imagem (desfaz a centragem da fase) ─────
    // Inverso do ifftshift aplicado em imaging_fft: traz o centro da faceta de
    // volta ao centro da sub-imagem. Garante o round-trip exato e mantém o
    // phase center na origem central (necessário para a correção de grid).
    fftshift_2d(work_grid);

    // ── Passo 3: Acumulação em δy ─────────────────────────────────────────────
    // O deslocamento de fase da faceta já foi desfeito no grid() (em função do
    // (u,v) real), então aqui basta acumular a parte REAL na região da faceta.
    // PARALELIZAÇÃO (OpenMP): cada (ix,iy) mapeia para um (gx,gy) exclusivo da
    // faceta → escrita sem corrida. (Facetas são regiões disjuntas da imagem e o
    // laço de facetas em main é serial, então não há sobreposição entre facetas.)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            // Posição no sky model completo
            int gx = facet.offset_x + ix;
            int gy = facet.offset_y + iy;

            // Boundary check
            if (gx < 0 || gx >= cfg.image_size_x ||
                gy < 0 || gy >= cfg.image_size_y) continue;

            // Acumula na imagem residual (parte real; δy é imagem real)
            state.delta_y(gx, gy) += work_grid(ix, iy).real();
        }
    }

    // Log
    std::cout << "    [imaging_fft_inv] Faceta " << facet_idx
              << ": grade UV → imagem δy acumulada.\n";
}

//=============================================================================
// HELPER: wrapping de índice (módulo positivo) para a periodicidade da DFT
//=============================================================================

static inline int wrap_index(int a, int n) {
    int r = a % n;
    return (r < 0) ? r + n : r;
}

//=============================================================================
// IMPLEMENTAÇÃO: degrid  —  v̂φi,MSj = Degrid(ĝφi,MSj, i, j)
//=============================================================================

void degrid(DDFacetState& state, int facet_idx, int ms_idx) {
    /**
     * Amostra a grade UV uniforme da faceta nas posições (u,v) reais das
     * visibilidades, convoluindo com um kernel Gaussiano (interpolação).
     * Acumula o resultado em predicted.data[k] (soma sobre facetas).
     */

    Facet& facet = state.facets[static_cast<size_t>(facet_idx)];
    const DDFacetConfig& cfg = state.config;
    MeasurementSetInfo& ms   = state.measurement_sets[static_cast<size_t>(ms_idx)];

    const VisibilitySet& vis = ms.visibilities;   // fornece (u, v, flag)

    const int    nx   = facet.npix_x;
    const int    ny   = facet.npix_y;
    const double cell = cfg.cell_size_rad;
    const int    W    = GRID_W_SUPPORT;
    const double s2   = 2.0 * GRID_KERNEL_SIGMA * GRID_KERNEL_SIGMA;

    // Saída no buffer PRÓPRIO da faceta (sem corrida entre facetas). O main soma
    // as contribuições de todas as facetas em ms.predicted depois do laço.
    if (facet.pred_contrib.size() != vis.nvis)
        facet.pred_contrib.assign(vis.nvis, std::complex<float>(0.0f, 0.0f));

    // PARALELIZAÇÃO (OpenMP): cada visibilidade k é independente — lê a grade
    // UV (somente leitura) e escreve pred_contrib[k] (índice exclusivo).
    // Sem corrida de escrita → laço embaraçosamente paralelo.
    const int    nvis = static_cast<int>(vis.nvis);
    const double l0   = facet.l_center;   // centro de fase da faceta
    const double m0   = facet.m_center;
    // Termo-w: n₀ = √(1 − l₀² − m₀²). A relação de Fourier no céu é esférica,
    // não plana — além de (u·l₀ + v·m₀) entra w·(n₀ − 1). Para a faceta central
    // (l₀=m₀=0) temos n₀=1 e o termo se anula. Guarda contra l₀²+m₀² ≥ 1.
    const double lm2  = l0 * l0 + m0 * m0;
    const double n0m1 = ddf_use_w_term() ? ((lm2 < 1.0) ? (std::sqrt(1.0 - lm2) - 1.0) : 0.0) : 0.0;
    // Ganho dependente da direção (DDE) desta faceta.
    const std::complex<double> gain = facet.directional_gain;
    long n_used = 0;
    #pragma omp parallel for schedule(static) reduction(+:n_used)
    for (int k = 0; k < nvis; ++k) {
        if (vis.flag[k]) continue;

        // (u,v) [wavelengths] → posição contínua em pixels na grade fftshift
        const double ix_c = vis.u[k] * nx * cell + nx / 2.0;
        const double iy_c = vis.v[k] * ny * cell + ny / 2.0;
        const int    ix0  = static_cast<int>(std::lround(ix_c));
        const int    iy0  = static_cast<int>(std::lround(iy_c));

        std::complex<double> acc(0.0, 0.0);
        double wsum = 0.0;

        for (int dj = -W; dj <= W; ++dj) {
            const int    iy = iy0 + dj;
            const double dv = static_cast<double>(iy) - iy_c;
            for (int di = -W; di <= W; ++di) {
                const int    ix = ix0 + di;
                const double du = static_cast<double>(ix) - ix_c;
                const double w  = std::exp(-(du * du + dv * dv) / s2);

                const std::complex<float>& g =
                    facet.uv_grid(wrap_index(ix, nx), wrap_index(iy, ny));
                acc  += std::complex<double>(g.real(), g.imag()) * w;
                wsum += w;
            }
        }

        if (wsum > 0.0) acc /= wsum;   // normaliza pela soma dos pesos

        // Deslocamento de fase da faceta, agora COM o termo-w:
        //   exp(-2πi(u·l₀ + v·m₀ + w·(n₀−1)))
        // Para a faceta central (l₀=m₀=0 → n₀=1) continua sendo identidade.
        const double ph = -2.0 * PI * (vis.u[k] * l0 + vis.v[k] * m0
                                       + vis.w[k] * n0m1);
        acc *= std::complex<double>(std::cos(ph), std::sin(ph));

        // DDE: aplica o ganho da direção desta faceta (identidade por padrão).
        acc *= gain;

        facet.pred_contrib[k] = std::complex<float>(static_cast<float>(acc.real()),
                                                    static_cast<float>(acc.imag()));
        ++n_used;
    }

    std::cout << "    [degrid] Faceta " << facet_idx
              << " (MS " << ms_idx << "): "
              << n_used << " visibilidades preditas (kernel "
              << (2 * W + 1) << "×" << (2 * W + 1) << ").\n";
}

//=============================================================================
// IMPLEMENTAÇÃO: grid  —  gφi,MSj = Grid(δvMSj, i, j)
//=============================================================================

void grid(DDFacetState& state, int facet_idx, int ms_idx) {
    /**
     * Operação adjunta do Degrid: espalha cada visibilidade residual δv_k na
     * grade UV da faceta, ponderada pelo kernel conjugado. ACUMULA em uv_grid.
     */

    Facet& facet = state.facets[static_cast<size_t>(facet_idx)];
    const DDFacetConfig& cfg = state.config;
    MeasurementSetInfo& ms   = state.measurement_sets[static_cast<size_t>(ms_idx)];

    const VisibilitySet& vis = ms.visibilities;   // fornece (u, v, flag)
    const VisibilitySet& res = ms.residual;        // δv (entrada)

    const int    nx   = facet.npix_x;
    const int    ny   = facet.npix_y;
    const double cell = cfg.cell_size_rad;
    const int    W    = GRID_W_SUPPORT;
    const double s2   = 2.0 * GRID_KERNEL_SIGMA * GRID_KERNEL_SIGMA;
    const int    nvis = static_cast<int>(vis.nvis);
    const double l0   = facet.l_center;   // centro de fase da faceta
    const double m0   = facet.m_center;
    // Termo-w (idêntico ao degrid; aqui entra com o sinal oposto na fase).
    const double lm2  = l0 * l0 + m0 * m0;
    const double n0m1 = ddf_use_w_term() ? ((lm2 < 1.0) ? (std::sqrt(1.0 - lm2) - 1.0) : 0.0) : 0.0;
    // DDE: no operador ADJUNTO aplica-se o conjugado do ganho.
    const std::complex<float> gain_conj =
        std::conj(std::complex<float>(static_cast<float>(facet.directional_gain.real()),
                                      static_cast<float>(facet.directional_gain.imag())));

    // PARALELIZAÇÃO (OpenMP): visibilidades vizinhas escrevem nas MESMAS células
    // da grade UV → corrida de escrita. Padrão de gridding paralelo: cada thread
    // acumula numa GRADE LOCAL própria e, ao final, soma (redução) na grade
    // compartilhada dentro de uma região crítica. Este é exatamente o padrão que
    // mapeia para um all-reduce MPI quando a paralelização for multi-nodal.
    long n_used = 0;
    #pragma omp parallel reduction(+:n_used)
    {
        std::vector<std::complex<float>> local(static_cast<size_t>(nx) * ny,
                                               std::complex<float>(0.0f, 0.0f));

        #pragma omp for schedule(static) nowait
        for (int k = 0; k < nvis; ++k) {
            if (vis.flag[k]) continue;

            // Deslocamento de fase da faceta (conjugado do degrid), com termo-w:
            // δv·exp(+2πi(u·l₀ + v·m₀ + w·(n₀−1))). Identidade na faceta central.
            const double ph = +2.0 * PI * (vis.u[k] * l0 + vis.v[k] * m0
                                           + vis.w[k] * n0m1);
            const std::complex<float> phase(static_cast<float>(std::cos(ph)),
                                            static_cast<float>(std::sin(ph)));
            // DDE adjunto: ×conj(G) — espelha o ×G aplicado no degrid.
            const std::complex<float> dv_k = res.data[k] * phase * gain_conj;

            const double ix_c = vis.u[k] * nx * cell + nx / 2.0;
            const double iy_c = vis.v[k] * ny * cell + ny / 2.0;
            const int    ix0  = static_cast<int>(std::lround(ix_c));
            const int    iy0  = static_cast<int>(std::lround(iy_c));

            for (int dj = -W; dj <= W; ++dj) {
                const int    iy = iy0 + dj;
                const double dv = static_cast<double>(iy) - iy_c;
                for (int di = -W; di <= W; ++di) {
                    const int    ix = ix0 + di;
                    const double du = static_cast<double>(ix) - ix_c;
                    const float  w  = static_cast<float>(std::exp(-(du * du + dv * dv) / s2));

                    // kernel real → conj(C) = C; mantido conj() por generalidade
                    const int gx = wrap_index(ix, nx), gy = wrap_index(iy, ny);
                    local[static_cast<size_t>(gy) * nx + gx] +=
                        dv_k * std::conj(std::complex<float>(w, 0.0f));
                }
            }
            ++n_used;
        }

        // Redução: soma a grade local desta thread no acumulador da faceta.
        // (grid_acc, separado do uv_grid usado pelo predict.)
        #pragma omp critical
        {
            for (size_t idx = 0; idx < local.size(); ++idx)
                facet.grid_acc.data[idx] += local[idx];
        }
    }

    std::cout << "    [grid] Faceta " << facet_idx
              << " (MS " << ms_idx << "): "
              << n_used << " visibilidades residuais espalhadas na grade UV.\n";
}

//=============================================================================
// IMPLEMENTAÇÃO: compute_residual  —  δvMSj = vMSj − v̂MSj
//=============================================================================

void compute_residual(DDFacetState& state, int ms_idx) {
    MeasurementSetInfo& ms = state.measurement_sets[static_cast<size_t>(ms_idx)];

    const VisibilitySet& meas = ms.visibilities;   // v
    const VisibilitySet& pred = ms.predicted;       // v̂
    VisibilitySet&       res  = ms.residual;        // δv (saída)

    if (res.nvis != meas.nvis) res.resize(meas.nvis);

    double res_energy = 0.0;
    for (size_t k = 0; k < meas.nvis; ++k) {
        res.u[k]    = meas.u[k];
        res.v[k]    = meas.v[k];
        res.w[k]    = meas.w[k];
        res.flag[k] = meas.flag[k];

        if (meas.flag[k]) {
            res.data[k] = std::complex<float>(0.0f, 0.0f);
            continue;
        }
        // δv = v − v̂  (Algoritmo 1, linha 30) → convenção convergente
        res.data[k] = meas.data[k] - pred.data[k];
        res_energy += std::norm(res.data[k]);
    }

    std::cout << "  [compute_residual] MS " << ms_idx
              << ": ‖δv‖² = " << std::scientific << std::setprecision(4)
              << res_energy << std::defaultfloat << "\n";
}

//=============================================================================
// IMPLEMENTAÇÃO: deconvolution  —  x̂ = Deconvolution(δy, PSF) — Högbom CLEAN
//=============================================================================

void deconvolution(DDFacetState& state) {
    const DDFacetConfig& cfg = state.config;
    ImageF&       dy  = state.delta_y;   // imagem residual (modificada in-place)
    ImageF&       x   = state.x;         // modelo do céu (atualizado)
    const ImageF& psf = state.psf;       // PSF, pico 1.0 no centro

    const int   nx  = dy.nx;
    const int   ny  = dy.ny;
    const int   pcx = psf.nx / 2;        // centro da PSF
    const int   pcy = psf.ny / 2;
    const float gain   = static_cast<float>(cfg.clean_gain);
    const float thresh = static_cast<float>(cfg.clean_threshold);

    int   n_comp   = 0;
    float peak_abs = 0.0f;

    // NOTA (OpenMP): o CLEAN é SEQUENCIAL — cada minor cycle depende da subtração
    // da PSF do pico anterior. Apenas a busca do pico e a subtração da PSF
    // (dentro de uma iteração) seriam paralelizáveis; para imagens pequenas o
    // custo de fork/join por iteração não compensa, então mantém-se serial.
    for (int it = 0; it < cfg.n_minor_iterations; ++it) {
        // ── Passo 1: encontra o pico em |δy| ────────────────────────────────
        int   px = 0, py = 0;
        float peak_val = 0.0f;
        float best_abs = -1.0f;
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const float a = std::fabs(dy(i, j));
                if (a > best_abs) {
                    best_abs = a;
                    peak_val = dy(i, j);
                    px = i; py = j;
                }
            }
        }
        peak_abs = best_abs;

        // ── Critério de parada por threshold ────────────────────────────────
        if (best_abs < thresh) {
            std::cout << "    [CLEAN] convergiu em " << it
                      << " iterações (|pico|=" << best_abs
                      << " < threshold=" << thresh << ").\n";
            break;
        }

        // ── Passo 2: adiciona componente ao modelo ──────────────────────────
        x(px, py) += gain * peak_val;

        // ── Passo 3: subtrai PSF escalonada de δy, centrada no pico ─────────
        for (int dj = 0; dj < psf.ny; ++dj) {
            const int iy = py + (dj - pcy);
            if (iy < 0 || iy >= ny) continue;
            for (int di = 0; di < psf.nx; ++di) {
                const int ix = px + (di - pcx);
                if (ix < 0 || ix >= nx) continue;
                dy(ix, iy) -= gain * peak_val * psf(di, dj);
            }
        }

        ++n_comp;
    }

    state.total_components += n_comp;
    state.peak_residual = peak_abs;

    std::cout << "    [CLEAN] " << n_comp << " componentes nesta deconvolução"
              << " | pico residual final = " << peak_abs << "\n";
}

} // namespace ddfacet

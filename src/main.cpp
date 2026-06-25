/**
 * =============================================================================
 * @file main.cpp
 * @brief DDFacet Sequential Imaging Pipeline — Demo v0.2
 *
 * IMPLEMENTADO NESTA VERSÃO:
 *   ✔  Initialization(v)           — ddfacet.cpp
 *   ✔  imaging_fft   (FFT step)    — ddfacet.cpp + fft.cpp
 *   ✔  imaging_fft_inv (IFFT step) — ddfacet.cpp + fft.cpp
 *
 * PENDENTE:
 *   ○  Degrid(ĝ, i, j)            — interpola grade UV → visibilidades
 *   ○  Grid(δv, i, j)             — acumula visibilidades → grade UV
 *   ○  Deconvolution(δy, PSF)     — MSMF CLEAN
 *   ○  Loop principal completo    — (k, j, i) conectando todas as funções
 *
 * ALGORITMO 1 (Monnier et al., SiPS IEEE 2022):
 * ─────────────────────────────────────────────
 * Initialization(v);
 * for k in K_MajorCycles do
 *   for j in J_MS do
 *     for i in I_Facets do
 *       ĝφi,MSj = FFT(x̂, i, j);          ← imaging_fft()    ✔ IMPL.
 *       v̂φi,MSj = Degrid(ĝφi,MSj, i, j); ← TODO
 *     end
 *     δvMSj = v̂MSj − vMSj;
 *     for i in I_Facets do
 *       gφi,MSj = Grid(δvMSj, i, j);      ← TODO
 *     end
 *   end
 *   for i in I_Facets do
 *     δy += FFT_inv(gφi, i, j);           ← imaging_fft_inv() ✔ IMPL.
 *   end
 *   x̂ = Deconvolution(δy, PSF);           ← TODO
 * end
 *
 * =============================================================================
 */

#include "ddfacet.h"
#include "ms_io.h"
#include "mpi_util.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>
#include <limits>
#include <chrono>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <fftw3.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace ddfacet;

// Estrutura auxiliar para descrever uma fonte pontual conhecida (para validação)
struct PointSource { int x, y; float amp; };

// ─── Helpers de teste ────────────────────────────────────────────────────────

/**
 * @brief Lê o sidecar de fontes verdadeiras (px py amp por linha) gerado por
 * tools/make_ms.py. Mantém o céu verdadeiro do C++ em sincronia com o MS.
 */
static std::vector<PointSource> read_sources_sidecar(const std::string& path) {
    std::vector<PointSource> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        int px, py; float amp;
        if (ss >> px >> py >> amp) out.push_back({ px, py, amp });
    }
    return out;
}

/** @brief Insere as fontes pontuais conhecidas na imagem (céu verdadeiro). */
static void inject_sources(ImageF& image, const std::vector<PointSource>& sources) {
    std::cout << "  Fontes verdadeiras:\n";
    for (const auto& src : sources) {
        if (src.x >= 0 && src.x < image.nx && src.y >= 0 && src.y < image.ny)
            image(src.x, src.y) = src.amp;
        std::cout << "    pix=(" << src.x << "," << src.y << ")  amp=" << src.amp << "\n";
    }
}

/**
 * @brief Constrói a PSF como o "dirty beam" da cobertura UV.
 *
 * O dirty beam é a resposta no domínio da imagem a uma fonte pontual central:
 * IFFT da função de amostragem (todas as visibilidades = 1), com o MESMO
 * kernel de gridding usado em grid(). Por ser produzido pela mesma cadeia de
 * operadores que a imagem suja, casa exatamente com a resposta de uma fonte
 * pontual → CLEAN consistente e estável. Recentrado com pico 1.0 em
 * (nx/2, ny/2). A cobertura hermitiana torna o beam real.
 *
 * `beam_peak_out` devolve o pico do beam ANTES da normalização (= S), o fator
 * de escala da imagem suja (uma fonte de fluxo A → pico ≈ A·T·S).
 */
static ImageF build_dirty_beam(const DDFacetState& state, int ms_idx,
                               int nx, int ny, double& beam_peak_out) {
    // nx,ny = TAMANHO DA FACETA (não da imagem): o dirty beam e a escala S
    // dependem da resolução de quem imageia. Para I=1, faceta = imagem inteira.
    (void)ms_idx;
    const double cell = state.config.cell_size_rad;
    const int    W    = GRID_W_SUPPORT;
    const double s2   = 2.0 * GRID_KERNEL_SIGMA * GRID_KERNEL_SIGMA;

    auto wrap = [](int a, int n) { int r = a % n; return (r < 0) ? r + n : r; };

    GridC beam(nx, ny);
    beam.zero();

    // Combina a cobertura UV de TODOS os MS deste rank (J>1). Sob MPI, cada rank
    // tem só o seu subconjunto; o all-reduce abaixo soma a cobertura de todos.
    for (const auto& ms : state.measurement_sets) {
        const VisibilitySet& vis = ms.visibilities;
        for (size_t k = 0; k < vis.nvis; ++k) {
            if (vis.flag[k]) continue;
            const double ix_c = vis.u[k] * nx * cell + nx / 2.0;
            const double iy_c = vis.v[k] * ny * cell + ny / 2.0;
            const int    ix0  = static_cast<int>(std::lround(ix_c));
            const int    iy0  = static_cast<int>(std::lround(iy_c));
            for (int dj = -W; dj <= W; ++dj) {
                const int iy = iy0 + dj; const double dv = iy - iy_c;
                for (int di = -W; di <= W; ++di) {
                    const int ix = ix0 + di; const double du = ix - ix_c;
                    const float w = static_cast<float>(std::exp(-(du * du + dv * dv) / s2));
                    beam(wrap(ix, nx), wrap(iy, ny)) += std::complex<float>(w, 0.0f);
                }
            }
        }
    }

    // MPI: soma a grade UV do beam entre todos os ranks (cobertura UV global).
    mpi_allreduce_floats(reinterpret_cast<float*>(beam.data.data()),
                         static_cast<size_t>(nx) * ny * 2);

    ifftshift_2d(beam);
    fft_2d(beam, /*forward=*/false);    // IFFT → imagem (dirty beam centrado em 0,0)

    // parte real + recentragem (fftshift) para pico no centro
    ImageF psf(nx, ny);
    const int sx = nx / 2, sy = ny / 2;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            psf((i + sx) % nx, (j + sy) % ny) = beam(i, j).real();

    // pico do dirty beam no centro = fator de escala S
    // (uma fonte de fluxo A produz pico ≈ A·T·S na imagem suja).
    const float pk = psf(nx / 2, ny / 2);
    beam_peak_out = static_cast<double>(pk);

    // normaliza pico = 1.0 no centro
    if (pk != 0.0f)
        for (auto& d : psf.data) d /= pk;

    return psf;
}

/**
 * @brief Calcula estatísticas de uma ImageF (min, max, soma, pico).
 */
static void print_image_stats(const ImageF& img, const std::string& name) {
    if (img.size() == 0) {
        std::cout << "  [" << name << "] vazia.\n";
        return;
    }
    float vmin = img.data[0], vmax = img.data[0];
    double vsum = 0.0;
    int peak_x = 0, peak_y = 0;

    for (int j = 0; j < img.ny; ++j) {
        for (int i = 0; i < img.nx; ++i) {
            float v = img(i, j);
            vsum += v;
            if (v < vmin) vmin = v;
            if (v > vmax) { vmax = v; peak_x = i; peak_y = j; }
        }
    }
    std::cout << "  [" << name << "] "
              << img.nx << "×" << img.ny
              << "  min=" << std::fixed << std::setprecision(4) << vmin
              << "  max=" << vmax
              << "  sum=" << vsum
              << "  peak=(" << peak_x << "," << peak_y << ")\n";
}

/**
 * @brief Valida o round-trip FFT → IFFT em um vetor sintético.
 *
 * Testa: IFFT(FFT(x)) ≈ x  (dentro de tolerância numérica)
 * Também verifica a propriedade de conjugação: FFT de sinal real é hermitiana.
 */
static void test_fft_roundtrip() {
    std::cout << "\n========================================\n";
    std::cout << "  TESTE: Round-trip FFT → IFFT (1D)\n";
    std::cout << "========================================\n";

    // Sinal de teste: combinação de senoides
    const size_t N = 64;
    std::vector<double> original(N);
    for (size_t i = 0; i < N; ++i) {
        original[i] = std::sin(2.0 * PI * 3.0 * i / N)   // freq 3
                    + 0.5 * std::cos(2.0 * PI * 7.0 * i / N);  // freq 7
    }

    // Copia para vetor complexo
    std::vector<std::complex<double>> data(N);
    for (size_t i = 0; i < N; ++i)
        data[i] = { original[i], 0.0 };

    // Forward FFT
    fft_1d(data, false);

    // Verifica simetria hermitiana: X[k] = conj(X[N-k]) para sinal real
    bool hermitian_ok = true;
    for (size_t k = 1; k < N / 2; ++k) {
        auto diff = std::abs(data[k] - std::conj(data[N - k]));
        if (diff > 1e-10) { hermitian_ok = false; break; }
    }
    std::cout << "  Simetria hermitiana: " << (hermitian_ok ? "OK" : "FALHOU") << "\n";

    // Inverse FFT
    fft_1d(data, true);

    // Compara com original
    double max_err = 0.0;
    for (size_t i = 0; i < N; ++i)
        max_err = std::max(max_err, std::abs(data[i].real() - original[i]));
    std::cout << "  Erro máx FFT→IFFT: " << std::scientific << max_err
              << "  " << (max_err < 1e-12 ? "✔ PASS" : "✘ FAIL") << "\n";
}

/**
 * @brief Valida o round-trip FFT_2D → IFFT_2D numa grade sintética.
 *
 * Cria uma grade com pontos conhecidos, aplica FFT 2D → fftshift → ifftshift → IFFT 2D
 * e verifica que a grade original é recuperada.
 */
static void test_fft2d_roundtrip() {
    std::cout << "\n========================================\n";
    std::cout << "  TESTE: Round-trip FFT2D → IFFT2D\n";
    std::cout << "========================================\n";

    const int N = 32;
    GridC grid(N, N);

    // Preenche com padrão sintético: ponto brilhante no centro
    grid(N/2, N/2) = {1.0f, 0.0f};
    grid(N/4, N/4) = {0.5f, 0.0f};

    // Salva original
    GridC original = grid;

    // Forward FFT + fftshift
    fft_2d(grid, true);
    fftshift_2d(grid);

    // ifftshift + Inverse FFT
    ifftshift_2d(grid);
    fft_2d(grid, false);

    // Compara com original
    float max_err = 0.0f;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            float err = std::abs(grid(i, j) - original(i, j));
            max_err = std::max(max_err, err);
        }
    }
    std::cout << "  Erro máx FFT2D→IFFT2D: " << std::scientific << max_err
              << "  " << (max_err < 1e-5f ? "✔ PASS" : "✘ FAIL") << "\n";
}

/**
 * @brief Função de correção de grid (apodização) no domínio da imagem.
 *
 * O kernel de convolução Gaussiano usado em grid()/degrid() equivale, no
 * domínio da imagem, a multiplicar por uma "taper" T(l,m) = FT do kernel.
 * Isto ATENUA fontes longe do phase center (centro da imagem). A correção de
 * grid divide a imagem por essa taper para restaurar as amplitudes corretas.
 *
 * Como o kernel é separável e simétrico, a taper 1-D é:
 *   g(i) = [ Σ_δ C(δ)·cos(2π·δ·(i − cx)/N) ] / Σ_δ C(δ)
 * e a taper 2-D é T(i,j) = g_x(i)·g_y(j), com pico 1.0 no centro.
 *
 * Retorna T (uma aplicação do kernel), com pico 1.0 no centro de CADA faceta.
 * Como cada faceta tem seu próprio phase center, a taper é "ladrilhada": a
 * taper de uma faceta (tamanho fnx×fny) é replicada por toda a imagem. Para
 * I=1 (faceta = imagem) reduz-se à taper de imagem inteira.
 */
static ImageF compute_grid_correction(int nx, int ny, int fnx, int fny) {
    const int    W  = GRID_W_SUPPORT;
    const double s2 = 2.0 * GRID_KERNEL_SIGMA * GRID_KERNEL_SIGMA;

    double sumC = 0.0;
    std::vector<double> C(2 * W + 1);
    for (int d = -W; d <= W; ++d) { C[d + W] = std::exp(-(d * d) / s2); sumC += C[d + W]; }

    auto taper1d = [&](int local, int n) {
        const double off = static_cast<double>(local) - n / 2.0;  // distância ao centro da faceta
        double acc = 0.0;
        for (int d = -W; d <= W; ++d)
            acc += C[d + W] * std::cos(2.0 * PI * d * off / n);
        return acc / sumC;
    };

    // taper local da faceta (replicada/ladrilhada na imagem)
    std::vector<double> gx(fnx), gy(fny);
    for (int i = 0; i < fnx; ++i) gx[i] = taper1d(i, fnx);
    for (int j = 0; j < fny; ++j) gy[j] = taper1d(j, fny);

    ImageF T(nx, ny);
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            T(i, j) = static_cast<float>(gx[i % fnx] * gy[j % fny]);
    return T;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    // Inicializa MPI (no-op sem -DUSE_MPI). Eixo J distribuído entre ranks.
    mpi_init(argc, argv);
    const int mpi_r = mpi_rank();
    const int mpi_n = mpi_size();
    const bool root = (mpi_r == 0);   // só o rank 0 imprime os blocos principais

    // O PLANEJADOR do FFTW NÃO é thread-safe: ao paralelizar o laço de facetas,
    // várias threads chamam fftw_plan_dft_2d concorrentemente. Esta chamada
    // instala um lock no planner (a execução já é thread-safe). Necessária para
    // I>1 com OpenMP. (Não exige modificar fft.cpp.)
    fftw_make_planner_thread_safe();

    if (root) {
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  DDFacet Imaging Pipeline (C++)          ║\n";
    std::cout << "║  faceamento + OpenMP + MPI + casacore    ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n\n";
    }

    // =========================================================================
    // TESTES UNITÁRIOS DO MÓDULO FFT (só no rank 0)
    // =========================================================================
    if (root) {
        test_fft_roundtrip();
        test_fft2d_roundtrip();
    }

    // =========================================================================
    // CONFIGURAÇÃO DO PIPELINE
    // =========================================================================
    if (root) {
    std::cout << "\n========================================\n";
    std::cout << "  CONFIGURAÇÃO DO PIPELINE\n";
    std::cout << "========================================\n";
    }

    DDFacetConfig config;
    config.n_major_cycles  = 10;     // K major cycles (convergência geométrica)
    // Nº de facetas por eixo configurável via DDF_FACETS (padrão 1).
    //   DDF_FACETS=2  → 2×2 = 4 facetas (demonstra o paralelismo por facetas).
    // A recuperação quantitativa é validada com I=1; com I>1 o eixo de facetas
    // paraleliza (grão grosso) e a fase de faceta exp(∓2πi(u·l₀+v·m₀)) já está
    // correta no degrid/grid.
    int nfac = 1;
    if (const char* e = std::getenv("DDF_FACETS")) { nfac = std::atoi(e); if (nfac < 1) nfac = 1; }
    config.n_facets_x      = nfac;
    config.n_facets_y      = nfac;
    config.image_size_x    = 128;    // Imagem 128×128 pixels
    config.image_size_y    = 128;
    config.cell_size_rad   = ARCSEC2RAD;
    config.ra0             = 266.4168 * DEG2RAD;   // Sgr A*
    config.dec0            = -29.0078 * DEG2RAD;
    config.n_minor_iterations = 20;  // limpeza gentil por ciclo (major/minor)
    config.clean_gain      = 0.1;
    config.clean_threshold = 1e-3;

    // =========================================================================
    // PASSO 1: INITIALIZATION(v)
    // =========================================================================
    std::cout << "\n";
    DDFacetState state;
    initialization(state, config);

    const int I = config.total_facets();

    // Nº GLOBAL de Measurement Sets (eixo J), configurável via DDF_NMS (padrão 1).
    // Vários MS = várias observações do mesmo céu (cobertura UV combinada).
    int J = 1;
    if (const char* e = std::getenv("DDF_NMS")) { J = std::atoi(e); if (J < 1) J = 1; }

    // =========================================================================
    // CÉU VERDADEIRO + MEASUREMENT SETS REAIS (casacore) + distribuição MPI
    // =========================================================================
    if (root) {
    std::cout << "\n========================================\n";
    std::cout << "  CÉU VERDADEIRO + MEASUREMENT SETS (casacore)\n";
    std::cout << "  J=" << J << " MS  |  MPI ranks=" << mpi_n << "\n";
    std::cout << "========================================\n";
    }

    const int N = config.n_facets_x;

    // O rank 0 GERA os J arquivos MS (cobertura UV distinta por MS); barreira
    // para os demais ranks esperarem antes de ler.
    auto ms_name = [&](int j) {
        return "data/sim_f" + std::to_string(N) + "_ms" + std::to_string(j) + ".ms";
    };
    // sidecar de fontes vem do MS 0 (make_ms.py escreve <ms_sem_ext>.sources.txt)
    const std::string src_path = "data/sim_f" + std::to_string(N) + "_ms0.sources.txt";
    if (root) {
        for (int j = 0; j < J; ++j) {
            if (!std::filesystem::exists(ms_name(j))) {
                std::cout << "  Gerando MS " << j << " (N=" << N << ")...\n";
                int rc = std::system(("python3 tools/make_ms.py \"" + ms_name(j) + "\" "
                            + std::to_string(N) + " " + std::to_string(j)).c_str());
                if (rc != 0) std::cout << "  [aviso] make_ms.py retornou " << rc << "\n";
            }
        }
    }
    mpi_barrier();   // todos esperam os MS existirem

    std::vector<PointSource> sources = read_sources_sidecar(src_path);
    if (sources.empty()) sources = { {56,56,1.0f}, {72,64,0.7f}, {64,72,0.4f} };
    inject_sources(state.x, sources);
    if (root) print_image_stats(state.x, "céu verdadeiro x_true");

    // DISTRIBUIÇÃO MPI DO EIXO J: o rank r processa os MS j com (j % size == r).
    // Cada rank guarda só o seu subconjunto em state.measurement_sets.
    // As visibilidades vêm SEMPRE do Measurement Set real via casacore (read_ms);
    // se a leitura falhar, é erro fatal (sem gerador sintético em memória).
    double ms_freq = 0.0;
    int n_local = 0;
    for (int j = 0; j < J; ++j) {
        if (j % mpi_n != mpi_r) continue;
        state.measurement_sets.emplace_back();
        MeasurementSetInfo& ms = state.measurement_sets.back();
        ms.id = j;
        if (!read_ms(ms_name(j), ms.visibilities, ms_freq)) {
            std::cerr << "[rank " << mpi_r << "] ERRO: não foi possível ler o MS '"
                      << ms_name(j) << "' via casacore. Abortando.\n";
            mpi_finalize();
            return 1;
        }
        ms.path = ms_name(j);
        ++n_local;
    }
    std::cout << "  [rank " << mpi_r << "] " << n_local
              << " MS locais (lidos via casacore).\n";

    // Taper de apodização, LADRILHADA por faceta (cada faceta tem seu phase
    // center). Usada só para recuperar o fluxo físico no final (x̂·T).
    const ImageF Tcorr = compute_grid_correction(config.image_size_x, config.image_size_y,
                                                  config.facet_size_x(), config.facet_size_y());
    if (root) print_image_stats(Tcorr, "taper de apodização T (por faceta)");

    // PSF = dirty beam NO TAMANHO DA FACETA (mesma resolução UV do imageamento
    // por faceta) → escala S e forma B corretas; mesmo beam para todas as
    // facetas (cobertura UV global). Inclui MPI_Allreduce da cobertura UV.
    double beam_peak = 1.0;
    state.psf = build_dirty_beam(state, /*ms_idx=*/0,
                                 config.facet_size_x(), config.facet_size_y(), beam_peak);
    if (root) {
        print_image_stats(state.psf, "PSF (dirty beam por faceta)");
        std::cout << "  Fator de escala da imagem suja S = " << beam_peak << "\n";
    }

    // =========================================================================
    // PONTO DE PARTIDA: modelo zerado — o pipeline deve RECUPERAR x_true
    // =========================================================================
    state.x.zero();
    state.total_components = 0;
    state.peak_residual    = 0.0;

    // =========================================================================
    // LOOP PRINCIPAL — Algoritmo 1 (Monnier et al., SiPS IEEE 2022)
    // =========================================================================
    if (root) {
    std::cout << "\n========================================\n";
    std::cout << "  LOOP PRINCIPAL (K=" << config.n_major_cycles
              << " major cycles, I=" << I << " faceta(s), J=" << J << " MS)\n";
#ifdef _OPENMP
    std::cout << "  OpenMP: " << omp_get_max_threads() << " thread(s)/rank";
#else
    std::cout << "  OpenMP: desabilitado";
#endif
#ifdef USE_MPI
    std::cout << "  |  MPI: " << mpi_n << " rank(s)";
#else
    std::cout << "  |  MPI: desabilitado (1 processo)";
#endif
    std::cout << "\n========================================\n";
    }

    const auto t_start = std::chrono::steady_clock::now();

    // Seleção do MELHOR modelo (early stopping): o CLEAN com PSF aproximada
    // (dispersão do kernel) acaba por super-ajustar; a energia do residual de
    // visibilidades ‖δv‖² atinge um mínimo e depois cresce. Guardamos o modelo
    // que produziu o menor ‖δv‖² e paramos se piorar em ciclos consecutivos.
    std::vector<double> resid_history;
    ImageF  best_model   = state.x;
    double  best_resid   = std::numeric_limits<double>::max();
    int     best_cycle   = 0;
    int     worse_streak = 0;

    for (int k = 0; k < config.n_major_cycles; ++k) {
        if (root) std::cout << "\n=== MAJOR CYCLE " << k << " ===\n";
        state.current_major_cycle = k;

        // Zera o acumulador do grid de todas as facetas (acumulação sobre MS).
        // O uv_grid (predict) é sobrescrito pelo imaging_fft a cada MS.
        for (auto& f : state.facets) f.grid_acc.zero();

        // Laço sobre os MS LOCAIS deste rank (sob MPI; todos se size=1).
        const int Jlocal = static_cast<int>(state.measurement_sets.size());
        double resid_energy = 0.0;
        for (int j = 0; j < Jlocal; ++j) {
            // --- PREDICT: ĝ = FFT(x̂); v̂ = Degrid(ĝ) ---
            // PARALELIZAÇÃO POR FACETAS (eixo I): cada faceta é independente e
            // escreve em seu próprio buffer (facet.pred_contrib) → sem corrida.
            // if(I>1): com I>1 paraleliza as facetas (grão grosso); com I=1 o
            // laço roda serial e o paralelismo interno dos operadores (vis/pixel)
            // é que atua. Evita aninhamento improdutivo.
            VisibilitySet& pred = state.measurement_sets[j].predicted;
            pred.resize(state.measurement_sets[j].visibilities.nvis);

            #pragma omp parallel for schedule(dynamic) if(I > 1)
            for (int i = 0; i < I; ++i) {
                imaging_fft(state, i, j);
                degrid(state, i, j);
            }

            // v̂ = Σ_i v̂_φi  (soma das contribuições das facetas)
            std::fill(pred.data.begin(), pred.data.end(),
                      std::complex<float>(0.0f, 0.0f));
            for (int i = 0; i < I; ++i) {
                const auto& pc = state.facets[static_cast<size_t>(i)].pred_contrib;
                const size_t n = std::min(pred.data.size(), pc.size());
                for (size_t kk = 0; kk < n; ++kk) pred.data[kk] += pc[kk];
            }

            // --- RESIDUAL: δv = v − v̂ ---
            compute_residual(state, j);
            for (const auto& dv : state.measurement_sets[j].residual.data)
                resid_energy += std::norm(dv);

            // --- GRID: g += Grid(δv) --- (paraleliza facetas; grades disjuntas)
            #pragma omp parallel for schedule(dynamic) if(I > 1)
            for (int i = 0; i < I; ++i) {
                grid(state, i, j);
            }
        }

        // === REDUÇÃO MPI (eixo J) ===========================================
        // g = Grid(δv) acumulado sobre TODOS os MS = soma das grades UV locais
        // de cada rank. MPI_Allreduce(SUM) dá a cada rank a grade UV global.
        // (Sem MPI é no-op: a grade local já é a global.)
        for (auto& f : state.facets)
            mpi_allreduce_floats(reinterpret_cast<float*>(f.grid_acc.data.data()),
                                 static_cast<size_t>(f.grid_acc.nx) * f.grid_acc.ny * 2);
        resid_energy = mpi_allreduce_double(resid_energy);
        resid_history.push_back(resid_energy);

        // Guarda o modelo (no início deste ciclo) que gerou o menor ‖δv‖².
        if (resid_energy < best_resid) {
            best_resid   = resid_energy;
            best_model   = state.x;
            best_cycle   = k;
            worse_streak = 0;
        } else {
            ++worse_streak;
        }

        // --- IMAGE: δy += FFT_inv(g) --- (paraleliza facetas; regiões disjuntas)
        state.delta_y.zero();
        #pragma omp parallel for schedule(dynamic) if(I > 1)
        for (int i = 0; i < I; ++i) {
            imaging_fft_inv(state, i);
        }
        // Escala da imagem suja por 1/S (uniforme → não desestabiliza o CLEAN).
        // Laço contrativo: o modelo x̂ converge para o fluxo APODIZADO a_true/T;
        // o fluxo físico é recuperado por x̂·T na validação final.
        if (beam_peak != 0.0)
            for (auto& d : state.delta_y.data)
                d = static_cast<float>(d / beam_peak);
        if (root) print_image_stats(state.delta_y, "δy (imagem residual suja)");

        // --- DECONVOLVE: x̂ = CLEAN(δy, PSF) ---
        deconvolution(state);

        if (root)
            std::cout << "  ‖δv‖² = " << std::scientific << std::setprecision(4) << resid_energy
                      << std::defaultfloat << "  | Components: " << state.total_components
                      << "  | Peak residual (img): " << std::scientific << std::setprecision(2)
                      << state.peak_residual << std::defaultfloat << "\n";

        // Early stopping: o residual de visibilidades piorou 2 ciclos seguidos.
        // (resid_energy é global via all-reduce → todos os ranks decidem igual.)
        if (worse_streak >= 2) {
            if (root) std::cout << "  [early stop] ‖δv‖² subindo — melhor modelo no ciclo "
                                << best_cycle << ".\n";
            break;
        }
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Restaura o melhor modelo encontrado.
    state.x = best_model;

    // Só o rank 0 reporta a validação e o resumo.
    if (root) {
    std::cout << "\n  Melhor modelo: ciclo " << best_cycle
              << "  (‖δv‖² = " << std::scientific << std::setprecision(4) << best_resid
              << std::defaultfloat << ")\n";
    std::cout << "  Tempo do loop principal: " << std::fixed << std::setprecision(1)
              << elapsed_ms << " ms"
#ifdef _OPENMP
              << "  (" << omp_get_max_threads() << " thread(s))"
#endif
              << std::defaultfloat << "\n";

    // =========================================================================
    // VALIDAÇÃO: recuperação das fontes
    // =========================================================================
    std::cout << "\n========================================\n";
    std::cout << "  VALIDAÇÃO: recuperação das fontes\n";
    std::cout << "========================================\n";

    // Modelo FÍSICO = x̂ · T (desfaz a apodização do laço de correção de grid).
    ImageF model_phys(state.x.nx, state.x.ny);
    for (int j = 0; j < state.x.ny; ++j)
        for (int i = 0; i < state.x.nx; ++i)
            model_phys(i, j) = state.x(i, j) * Tcorr(i, j);

    print_image_stats(state.x,    "modelo recuperado x̂ (unidades apodizadas)");
    print_image_stats(model_phys, "modelo físico x̂·T");

    // (a) O residual de visibilidades deve ter caído do 1º ciclo até o melhor.
    const bool residual_decreasing =
        (resid_history.size() >= 2) && (best_resid < resid_history.front() * 0.5);
    std::cout << "\n  Histórico de ‖δv‖² (primeiro→último):\n    ";
    for (double p : resid_history) std::cout << std::scientific << std::setprecision(2) << p << "  ";
    std::cout << std::defaultfloat << "\n  Pico residual decrescente: "
              << std::defaultfloat << "\n  ‖δv‖² convergiu (< 50% do inicial): "
              << (residual_decreasing ? "✔ SIM" : "✘ NÃO") << "\n";

    // (b) Fluxo físico recuperado em janela 3×3 ao redor de cada fonte
    std::cout << "\n  Fluxo físico recuperado por fonte (janela 3×3):\n";
    bool all_recovered = true;
    for (const auto& src : sources) {
        double win_flux = 0.0;
        for (int dj = -1; dj <= 1; ++dj)
            for (int di = -1; di <= 1; ++di) {
                int ix = src.x + di, iy = src.y + dj;
                if (ix >= 0 && ix < model_phys.nx && iy >= 0 && iy < model_phys.ny)
                    win_flux += model_phys(ix, iy);
            }
        const double rel = (src.amp > 0) ? win_flux / src.amp : 0.0;
        const bool ok = rel > 0.6 && rel < 1.4;   // tolerância ±40%
        all_recovered = all_recovered && ok;
        std::cout << "    pix=(" << src.x << "," << src.y << ")"
                  << "  amp_true=" << std::fixed << std::setprecision(3) << src.amp
                  << "  recuperado=" << win_flux
                  << "  (" << std::setprecision(0) << rel * 100.0 << "%)  "
                  << (ok ? "✔" : "✘") << std::defaultfloat << "\n";
    }

    // (c) Pico global do modelo físico deve coincidir com a fonte mais brilhante
    int mx = 0, my = 0; float mval = model_phys.data[0];
    for (int j = 0; j < model_phys.ny; ++j)
        for (int i = 0; i < model_phys.nx; ++i)
            if (model_phys(i, j) > mval) { mval = model_phys(i, j); mx = i; my = j; }
    const PointSource& brightest = sources[0];
    const int dist = std::abs(mx - brightest.x) + std::abs(my - brightest.y);
    std::cout << "\n  Pico do modelo em (" << mx << "," << my << "), fonte mais"
              << " brilhante em (" << brightest.x << "," << brightest.y << ")"
              << "  → distância L1 = " << dist << " px  "
              << (dist <= 1 ? "✔" : "✘") << "\n";

    std::cout << "\n  RESULTADO: "
              << ((residual_decreasing && all_recovered && dist <= 1)
                      ? "✔ Fontes recuperadas com sucesso."
                      : "✘ Recuperação fora da tolerância.")
              << "\n";

    // =========================================================================
    // RESUMO DO ESTADO
    // =========================================================================
    std::cout << "\n";
    print_state_info(state);

    // =========================================================================
    // PRÓXIMOS PASSOS
    // =========================================================================
    std::cout << "========================================\n";
    std::cout << "  PRÓXIMOS PASSOS\n";
    std::cout << "========================================\n\n";
    std::cout << "  ✔ [FEITO]   FFT(x̂, i, j)           → imaging_fft()\n";
    std::cout << "  ✔ [FEITO]   FFT_inv(gφi, i, j)     → imaging_fft_inv()\n";
    std::cout << "  ✔ [FEITO]   Degrid(ĝ, i, j)         → degrid()  (kernel Gaussiano)\n";
    std::cout << "  ✔ [FEITO]   Grid(δv, i, j)          → grid()    (adjunto)\n";
    std::cout << "  ✔ [FEITO]   compute_residual         → δv = v − v̂\n";
    std::cout << "  ✔ [FEITO]   Deconvolution(δy, PSF)  → Högbom CLEAN\n";
    std::cout << "  ✔ [FEITO]   Loop principal (k,j,i)  → Algoritmo 1 completo\n";
    std::cout << "  ✔ [FEITO]   Faceamento I>1 end-to-end (dirty beam/escala por faceta)\n";
    std::cout << "  ✔ [FEITO]   Paralelização OpenMP (facetas + vis/pixels)\n";
    std::cout << "  ✔ [FEITO]   Leitura de MS real (casacore)\n";
    std::cout << "  ✔ [FEITO]   MPI eixo J (all-reduce das grades UV)\n";
    std::cout << "  ○ [FASE 3]  W-term, MSMF (multi-frequência), overlap de facetas\n";
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════╗\n";
    std::cout << "║  PROGRAMA FINALIZADO COM SUCESSO         ║\n";
    std::cout << "╚══════════════════════════════════════════╝\n";
    }   // if (root)

    mpi_finalize();
    return 0;
}

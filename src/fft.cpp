/**
 * @file fft.cpp
 * @brief FFT/IFFT 2D via FFTW3 (double precision)
 *
 * Dependência: libfftw3-dev
 *   Ubuntu/WSL:  sudo apt install libfftw3-dev
 *   Fedora:      sudo dnf install fftw-devel
 *   macOS:       brew install fftw
 */

#include <fftw3.h>

#include "ddfacet.h"
#include <cmath>
#include <stdexcept>
#include <iostream>

namespace ddfacet {

// ─── Helpers internos ─────────────────────────────────────────────────────────

static inline bool is_pow2(size_t n) { return n > 0 && (n & (n - 1)) == 0; }

size_t next_pow2(size_t n) {
    if (n == 0) return 1;
    if (is_pow2(n)) return n;
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// ─── fft_1d ───────────────────────────────────────────────────────────────────
// Usado nos testes unitários e como utilitário interno.
// FFTW_BACKWARD não normaliza → divisão por N aplicada manualmente.

void fft_1d(std::vector<std::complex<double>>& data, bool inverse) {
    const int n = static_cast<int>(data.size());
    if (n == 0) return;

    // Aloca buffer alinhado FFTW3
    fftw_complex* buf = static_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * static_cast<size_t>(n)));

    for (int i = 0; i < n; ++i) {
        buf[i][0] = data[static_cast<size_t>(i)].real();
        buf[i][1] = data[static_cast<size_t>(i)].imag();
    }

    const int sign = inverse ? FFTW_BACKWARD : FFTW_FORWARD;
    fftw_plan p = fftw_plan_dft_1d(n, buf, buf, sign, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    const double norm = inverse ? 1.0 / static_cast<double>(n) : 1.0;
    for (int i = 0; i < n; ++i) {
        data[static_cast<size_t>(i)] = {
            buf[i][0] * norm,
            buf[i][1] * norm
        };
    }

    fftw_free(buf);
}

// ─── fft_2d ───────────────────────────────────────────────────────────────────
// GridC usa complex<float>, layout row-major: grid(i,j) = data[j*nx + i]
// FFTW espera layout row-major com n0=ny (linhas) e n1=nx (colunas).
// Portanto o mapeamento é direto: buf[j*nx + i] ↔ grid(i,j).
//
// Para FFTW_BACKWARD (IFFT): normalização 1/(nx*ny) aplicada manualmente.

void fft_2d(GridC& grid, bool forward) {
    const int nx = grid.nx;
    const int ny = grid.ny;
    const size_t n = static_cast<size_t>(nx * ny);

    // Aloca buffer double alinhado
    fftw_complex* buf = static_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * n));

    // Copia GridC (float) → buffer FFTW3 (double)
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const size_t idx = static_cast<size_t>(j * nx + i);
            buf[idx][0] = static_cast<double>(grid(i, j).real());
            buf[idx][1] = static_cast<double>(grid(i, j).imag());
        }
    }

    // Cria plano e executa
    // FFTW_ESTIMATE: sem benchmark; rápido para criar. Usar FFTW_MEASURE em produção.
    const int sign = forward ? FFTW_FORWARD : FFTW_BACKWARD;
    fftw_plan p = fftw_plan_dft_2d(ny, nx, buf, buf, sign, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    // Copia de volta com normalização para IFFT
    const double norm = forward ? 1.0 : 1.0 / static_cast<double>(nx * ny);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const size_t idx = static_cast<size_t>(j * nx + i);
            grid(i, j) = {
                static_cast<float>(buf[idx][0] * norm),
                static_cast<float>(buf[idx][1] * norm)
            };
        }
    }

    fftw_free(buf);
}

// ─── fftshift_2d / ifftshift_2d ──────────────────────────────────────────────

void fftshift_2d(GridC& grid) {
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int sx = nx / 2;
    const int sy = ny / 2;
    GridC tmp = grid;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            grid((i + sx) % nx, (j + sy) % ny) = tmp(i, j);
}

void ifftshift_2d(GridC& grid) {
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int sx = (nx + 1) / 2;
    const int sy = (ny + 1) / 2;
    GridC tmp = grid;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            grid((i + sx) % nx, (j + sy) % ny) = tmp(i, j);
}

} // namespace ddfacet

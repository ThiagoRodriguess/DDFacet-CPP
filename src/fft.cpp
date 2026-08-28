/**
 * @file fft.cpp
 * @brief Iterative radix-2 FFT and fftshift (see fft.h).
 */
#include "fft.h"

#include <cmath>
#include <cstddef>

namespace ddfacet {

namespace {

constexpr double PI = 3.14159265358979323846;

/** @brief 1-D radix-2 FFT, in place. */
void fft1d(float* re, float* im, int n, int sign) {
    // Bit-reversal permutation.
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            float t = re[i]; re[i] = re[j]; re[j] = t;
            t = im[i]; im[i] = im[j]; im[j] = t;
        }
    }

    // Butterflies, doubling the block length each pass.
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = sign * 2.0 * PI / len;
        const double wr = std::cos(ang), wi = std::sin(ang);
        for (int i = 0; i < n; i += len) {
            double cr = 1.0, ci = 0.0;
            for (int k = 0; k < len / 2; ++k) {
                const int a = i + k, b = i + k + len / 2;
                const double xr = re[b] * cr - im[b] * ci;
                const double xi = re[b] * ci + im[b] * cr;
                re[b] = static_cast<float>(re[a] - xr);
                im[b] = static_cast<float>(im[a] - xi);
                re[a] = static_cast<float>(re[a] + xr);
                im[a] = static_cast<float>(im[a] + xi);
                const double ncr = cr * wr - ci * wi;
                ci = cr * wi + ci * wr;
                cr = ncr;
            }
        }
    }

    if (sign > 0)
        for (int i = 0; i < n; ++i) { re[i] /= n; im[i] /= n; }
}

} // namespace

void fft2d(std::vector<float>& re, std::vector<float>& im,
           int nx, int ny, FFTDirection dir) {
    const int sign = static_cast<int>(dir);

    std::vector<float> row_re(nx), row_im(nx);
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            row_re[i] = re[static_cast<std::size_t>(j) * nx + i];
            row_im[i] = im[static_cast<std::size_t>(j) * nx + i];
        }
        fft1d(row_re.data(), row_im.data(), nx, sign);
        for (int i = 0; i < nx; ++i) {
            re[static_cast<std::size_t>(j) * nx + i] = row_re[i];
            im[static_cast<std::size_t>(j) * nx + i] = row_im[i];
        }
    }

    std::vector<float> col_re(ny), col_im(ny);
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            col_re[j] = re[static_cast<std::size_t>(j) * nx + i];
            col_im[j] = im[static_cast<std::size_t>(j) * nx + i];
        }
        fft1d(col_re.data(), col_im.data(), ny, sign);
        for (int j = 0; j < ny; ++j) {
            re[static_cast<std::size_t>(j) * nx + i] = col_re[j];
            im[static_cast<std::size_t>(j) * nx + i] = col_im[j];
        }
    }
}

void fftshift(std::vector<float>& a, int nx, int ny) {
    std::vector<float> t(a.size());
    const int hx = nx / 2, hy = ny / 2;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            t[static_cast<std::size_t>((j + hy) % ny) * nx + (i + hx) % nx] =
                a[static_cast<std::size_t>(j) * nx + i];
    a.swap(t);
}

} // namespace ddfacet

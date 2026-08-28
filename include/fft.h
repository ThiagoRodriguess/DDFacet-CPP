/**
 * @file fft.h
 * @brief In-tree FFT — no FFTW.
 *
 * FFTW cannot be assumed inside the OpenMP Cluster container, so the transform
 * is implemented here. It is an iterative radix-2 Cooley-Tukey, which is why
 * every image dimension must be a power of two.
 *
 * Complex data is carried as two parallel float arrays (real, imaginary), the
 * same representation used across the offload boundary.
 */
#ifndef FFT_H
#define FFT_H

#include <vector>

namespace ddfacet {

/** Sign convention for the transforms below. */
enum FFTDirection {
    FFT_FORWARD = -1,   ///< image -> UV
    FFT_INVERSE = +1    ///< UV -> image (includes the 1/N normalisation)
};

/**
 * @brief Separable 2-D FFT, in place: rows first, then columns.
 *
 * @param re,im  nx*ny elements, row-major
 * @param nx,ny  dimensions; both must be powers of two
 * @param dir    FFT_FORWARD or FFT_INVERSE
 */
void fft2d(std::vector<float>& re, std::vector<float>& im,
           int nx, int ny, FFTDirection dir);

/**
 * @brief Swap quadrants, moving the origin between corner and centre.
 *
 * Applied before and after each transform so that the zero frequency sits in
 * the middle of the grid, which is what the gridding code assumes.
 */
void fftshift(std::vector<float>& a, int nx, int ny);

/** @brief True when n is a power of two (the radix-2 requirement). */
inline bool is_power_of_two(int n) { return n > 0 && (n & (n - 1)) == 0; }

} // namespace ddfacet

#endif // FFT_H

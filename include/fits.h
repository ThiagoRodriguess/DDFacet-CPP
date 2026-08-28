/**
 * @file fits.h
 * @brief Minimal FITS writer — no cfitsio.
 *
 * Enough of the format to write a 2-D single-precision image that DS9, astropy
 * and casacore can all open. Written by hand because the OpenMP Cluster
 * container carries no FITS library.
 */
#ifndef FITS_H
#define FITS_H

#include <string>
#include <vector>

namespace ddfacet {

/**
 * @brief Write a 2-D float image as FITS (BITPIX=-32, big-endian).
 *
 * FITS stores floats big-endian regardless of the host, so the bytes are
 * swapped explicitly on the way out.
 *
 * @param path   output file
 * @param img    nx*ny pixels, row-major
 * @param nx,ny  image dimensions
 * @return false if the file could not be written
 */
bool write_fits(const std::string& path, const std::vector<float>& img,
                int nx, int ny);

} // namespace ddfacet

#endif // FITS_H

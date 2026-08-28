/**
 * @file fits.cpp
 * @brief Minimal FITS writer (see fits.h).
 */
#include "fits.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace ddfacet {

namespace {

constexpr int BLOCK  = 2880;   // FITS block size, in bytes
constexpr int CARD   = 80;     // header card size
constexpr int CARDS_PER_BLOCK = BLOCK / CARD;

} // namespace

bool write_fits(const std::string& path, const std::vector<float>& img,
                int nx, int ny) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;

    // -- Header: 80-character cards, padded to a whole 2880-byte block -------
    int ncards = 0;
    auto card = [&](const char* text) {
        char buf[CARD];
        std::memset(buf, ' ', CARD);
        std::size_t len = std::strlen(text);
        if (len > CARD) len = CARD;
        std::memcpy(buf, text, len);
        std::fwrite(buf, 1, CARD, f);
        ++ncards;
    };

    char tmp[CARD + 1];
    card("SIMPLE  =                    T");
    card("BITPIX  =                  -32");
    card("NAXIS   =                    2");
    std::snprintf(tmp, sizeof(tmp), "NAXIS1  = %20d", nx); card(tmp);
    std::snprintf(tmp, sizeof(tmp), "NAXIS2  = %20d", ny); card(tmp);
    card("BSCALE  =                  1.0");
    card("BZERO   =                  0.0");
    card("END");
    while (ncards % CARDS_PER_BLOCK != 0) card("");

    // -- Data: float32, big-endian, x varying fastest ------------------------
    std::size_t nbytes = 0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const float val = img[static_cast<std::size_t>(j) * nx + i];
            std::uint32_t u;
            std::memcpy(&u, &val, 4);
            const unsigned char be[4] = {
                static_cast<unsigned char>((u >> 24) & 0xFF),
                static_cast<unsigned char>((u >> 16) & 0xFF),
                static_cast<unsigned char>((u >>  8) & 0xFF),
                static_cast<unsigned char>( u        & 0xFF)
            };
            std::fwrite(be, 1, 4, f);
            nbytes += 4;
        }
    }
    while (nbytes % BLOCK != 0) { const char z = 0; std::fwrite(&z, 1, 1, f); ++nbytes; }

    const bool ok = (std::ferror(f) == 0);
    std::fclose(f);
    return ok;
}

} // namespace ddfacet

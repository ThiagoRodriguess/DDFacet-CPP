/**
 * @file vis_file.cpp
 * @brief Reader and writer for the .vis format (see vis_file.h).
 *
 * Deliberately free of dependencies — this file is linked into the imaging
 * binary, which must build inside the OMPC container.
 */
#include "vis_file.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace ddfacet {

namespace {
constexpr char   MAGIC[8]    = {'D','D','F','V','I','S','0','1'};
constexpr int    N_ARRAYS    = 6;   // u, v, w, re, im, flag
} // namespace

void VisFile::resize(int n) {
    nvis = n;
    u.resize(n); v.resize(n); w.resize(n);
    re.resize(n); im.resize(n); flag.resize(n);
}

bool vis_file_read(const std::string& path, VisFile& out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "[vis_file] could not open '%s'\n", path.c_str());
        return false;
    }

    char magic[8] = {0};
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, MAGIC, 8) != 0) {
        std::fprintf(stderr, "[vis_file] '%s' is not a DDFVIS01 file\n", path.c_str());
        std::fclose(f);
        return false;
    }

    std::int32_t nvis = 0, channel = 0;
    std::size_t head = 0;
    head += std::fread(&nvis,        sizeof(std::int32_t), 1, f);
    head += std::fread(&channel,     sizeof(std::int32_t), 1, f);
    head += std::fread(&out.freq_hz, sizeof(double),       1, f);
    head += std::fread(&out.umax_wl, sizeof(double),       1, f);
    if (head != 4 || nvis <= 0) {
        std::fprintf(stderr, "[vis_file] invalid header in '%s' (nvis=%d)\n",
                     path.c_str(), nvis);
        std::fclose(f);
        return false;
    }

    out.channel = channel;
    out.resize(nvis);

    const std::size_t n = static_cast<std::size_t>(nvis);
    std::size_t got = 0;
    got += std::fread(out.u.data(),    sizeof(double), n, f);
    got += std::fread(out.v.data(),    sizeof(double), n, f);
    got += std::fread(out.w.data(),    sizeof(double), n, f);
    got += std::fread(out.re.data(),   sizeof(float),  n, f);
    got += std::fread(out.im.data(),   sizeof(float),  n, f);
    got += std::fread(out.flag.data(), sizeof(unsigned char), n, f);

    const bool ok = (got == N_ARRAYS * n) && (std::ferror(f) == 0);
    if (!ok)
        std::fprintf(stderr, "[vis_file] '%s' truncated: read %zu of %zu elements\n",
                     path.c_str(), got, N_ARRAYS * n);
    std::fclose(f);
    return ok;
}

bool vis_file_write(const std::string& path, const VisFile& in) {
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "[vis_file] could not open '%s' for writing\n", path.c_str());
        return false;
    }

    const std::int32_t nvis    = static_cast<std::int32_t>(in.nvis);
    const std::int32_t channel = static_cast<std::int32_t>(in.channel);
    const std::size_t  n       = static_cast<std::size_t>(in.nvis);

    std::fwrite(MAGIC, 1, 8, f);
    std::fwrite(&nvis,       sizeof(std::int32_t), 1, f);
    std::fwrite(&channel,    sizeof(std::int32_t), 1, f);
    std::fwrite(&in.freq_hz, sizeof(double),       1, f);
    std::fwrite(&in.umax_wl, sizeof(double),       1, f);

    std::fwrite(in.u.data(),    sizeof(double), n, f);
    std::fwrite(in.v.data(),    sizeof(double), n, f);
    std::fwrite(in.w.data(),    sizeof(double), n, f);
    std::fwrite(in.re.data(),   sizeof(float),  n, f);
    std::fwrite(in.im.data(),   sizeof(float),  n, f);
    std::fwrite(in.flag.data(), sizeof(unsigned char), n, f);

    const bool ok = (std::ferror(f) == 0);
    std::fclose(f);
    return ok;
}

} // namespace ddfacet

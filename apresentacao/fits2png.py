#!/usr/bin/env python3
"""Le um FITS (BITPIX=-32 big-endian, escrito pelo nosso write_fits) e gera PNG
colormapeado, sem depender de matplotlib/PIL/astropy. Apenas numpy + zlib.

Uso: python3 fits2png.py <in.fits> <out.png> [--stretch asinh|linear] [--cmap inferno|viridis]
"""
import sys, zlib, struct
import numpy as np

def read_fits(path):
    with open(path, "rb") as f:
        raw = f.read()
    # cabecalho: blocos de 2880 bytes, cards de 80 chars, ate "END"
    hdr = {}
    off = 0
    end = False
    while not end:
        block = raw[off:off+2880]; off += 2880
        for i in range(0, 2880, 80):
            card = block[i:i+80].decode("ascii", "replace")
            key = card[:8].strip()
            if key == "END":
                end = True; break
            if "=" in card:
                val = card[10:].split("/")[0].strip()
                hdr[key] = val
    naxis1 = int(hdr["NAXIS1"]); naxis2 = int(hdr["NAXIS2"])
    bitpix = int(hdr["BITPIX"])
    assert bitpix == -32, f"esperado BITPIX=-32, veio {bitpix}"
    n = naxis1 * naxis2
    data = np.frombuffer(raw[off:off+4*n], dtype=">f4").astype(np.float64)
    data = data.reshape((naxis2, naxis1))
    return data, hdr

# ---- colormaps (poucos pontos de controle, interpolados) ----
CMAPS = {
    "inferno": [(0,0,0),(0.3,0.05,0.35),(0.6,0.13,0.35),(0.87,0.29,0.15),
                (0.99,0.62,0.0),(0.98,0.9,0.35),(0.99,1.0,0.9)],
    "viridis": [(0.27,0.0,0.33),(0.28,0.17,0.47),(0.23,0.32,0.55),(0.17,0.45,0.56),
                (0.13,0.57,0.55),(0.21,0.72,0.47),(0.57,0.85,0.27),(0.99,0.91,0.14)],
    "gray":    [(0,0,0),(1,1,1)],
}

def apply_cmap(t, name):
    pts = CMAPS[name]; m = len(pts)-1
    x = t*m
    lo = np.clip(np.floor(x).astype(int),0,m-1)
    fr = (x-lo)[...,None]
    pa = np.array(pts)
    rgb = pa[lo]*(1-fr) + pa[np.clip(lo+1,0,m)]*fr
    return (np.clip(rgb,0,1)*255).astype(np.uint8)

def write_png(path, rgb):
    h,w,_ = rgb.shape
    def chunk(typ,data):
        c = typ+data
        return struct.pack(">I",len(data))+c+struct.pack(">I",zlib.crc32(c)&0xffffffff)
    raw = b"".join(b"\x00"+rgb[y].tobytes() for y in range(h))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB",w,h,8,2,0,0,0))
    png += chunk(b"IDAT", zlib.compress(raw,9))
    png += chunk(b"IEND", b"")
    with open(path,"wb") as f: f.write(png)

def main():
    inp, out = sys.argv[1], sys.argv[2]
    stretch = "asinh"; cmap = "inferno"
    for a in sys.argv[3:]:
        if a.startswith("--stretch"): stretch = sys.argv[sys.argv.index(a)+1]
        if a.startswith("--cmap"):    cmap = sys.argv[sys.argv.index(a)+1]
    d, hdr = read_fits(inp)
    vmin = float(d.min()); vmax = float(d.max()); vsum = float(d.sum())
    pk = np.unravel_index(np.argmax(d), d.shape)
    # normaliza [0, vmax] (dirty image tipicamente >=0)
    lo = max(vmin, 0.0)
    dd = np.clip((d - lo)/(vmax - lo + 1e-30), 0, 1)
    if stretch == "asinh":
        a = 0.08
        dd = np.arcsinh(dd/a)/np.arcsinh(1.0/a)
    rgb = apply_cmap(dd, cmap)
    # eixo y para cima (FITS convention): flip
    rgb = rgb[::-1]
    write_png(out, rgb)
    print(f"{inp}: {d.shape[1]}x{d.shape[0]} min={vmin:.5f} max={vmax:.5f} "
          f"sum={vsum:.4f} peak=({pk[1]},{pk[0]}) -> {out}")

if __name__ == "__main__":
    main()

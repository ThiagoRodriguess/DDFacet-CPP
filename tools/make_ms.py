#!/usr/bin/env python3
"""
Gera um Measurement Set (MS) REAL usando python-casacore, a partir de fontes
pontuais conhecidas. Substitui o gerador "sintético" em memória — agora os
dados passam por um arquivo MS de verdade (formato CASA), lido depois pelo C++.

IMPORTANTE: os parâmetros abaixo DEVEM casar com a DDFacetConfig em src/main.cpp
(tamanho da imagem, cell_size, fontes), senão a validação de recuperação falha.

Uso:  python3 tools/make_ms.py data/sim.ms
"""
import sys, os, shutil
import numpy as np
from casacore.tables import default_ms, table, makearrcoldesc, maketabdesc

# ---- parâmetros (CASAR com o C++) -------------------------------------------
NX = NY   = 128
ARCSEC    = np.pi / 180.0 / 3600.0
CELL      = ARCSEC                 # cell_size_rad
FREQ      = 1.4e9                  # Hz (banda L)
C_LIGHT   = 299792458.0
WL        = C_LIGHT / FREQ         # comprimento de onda [m]
N_PAIRS   = 8000                   # → 16000 visibilidades (pares hermitianos)
U_MAX_WL  = 0.45 / CELL            # raio do disco UV [wavelengths]
SEED      = 20240501

def sources_for(nfac):
    """Fontes (px,py,amp) escolhidas conforme o nº de facetas por eixo.
    - nfac=1: perto do centro da imagem (campo útil da faceta única).
    - nfac>=2: nos CENTROS de facetas distintas (campo bem-imageado por faceta)."""
    if nfac <= 1:
        return [(56, 56, 1.0), (72, 64, 0.7), (64, 72, 0.4)]
    f = NX // nfac                       # tamanho da faceta
    def center(fi, fj):
        return (fi * f + f // 2, fj * f + f // 2)
    amps = [1.0, 0.7, 0.4]
    # facetas distintas: (0,0), (nfac-1,nfac-1), (nfac-1,0)
    fac = [(0, 0), (nfac - 1, nfac - 1), (nfac - 1, 0)]
    out = []
    for (fi, fj), a in zip(fac, amps):
        px, py = center(fi, fj)
        out.append((px, py, a))
    return out

def main(ms_path, nfac=1, ms_index=0):
    # cada MS (índice) usa uma semente diferente → cobertura UV distinta
    # (observações independentes do MESMO céu). Combinar J MS melhora a imagem.
    rng = np.random.default_rng(SEED + ms_index * 7919)
    SOURCES = sources_for(nfac)

    # cobertura UV (wavelengths) em pares hermitianos (u,v) e (-u,-v)
    us, vs = [], []
    while len(us) < N_PAIRS:
        u = (rng.random() * 2 - 1) * U_MAX_WL
        v = (rng.random() * 2 - 1) * U_MAX_WL
        if u * u + v * v <= U_MAX_WL * U_MAX_WL:
            us.append(u); vs.append(v)
    us = np.array(us); vs = np.array(vs)
    u_all = np.concatenate([us, -us])
    v_all = np.concatenate([vs, -vs])
    w_all = np.zeros_like(u_all)
    n = u_all.size

    # DATA = DFT direta das fontes (equação de medida)
    #   V(u,v) = Σ A·exp(-2πi (u·l + v·m)),  l=(px-NX/2)·cell
    data = np.zeros(n, dtype=np.complex128)
    for (px, py, amp) in SOURCES:
        l = (px - NX / 2.0) * CELL
        m = (py - NY / 2.0) * CELL
        data += amp * np.exp(-2j * np.pi * (u_all * l + v_all * m))

    uvw_m = np.column_stack([u_all, v_all, w_all]) * WL   # MS guarda UVW em METROS

    # ---- cria o MS ----------------------------------------------------------
    if os.path.exists(ms_path):
        shutil.rmtree(ms_path)
    os.makedirs(os.path.dirname(ms_path) or '.', exist_ok=True)
    default_ms(ms_path)

    t = table(ms_path, readonly=False, ack=False)
    t.addrows(n)
    # a coluna DATA (complexa, [nchan, npol]) é opcional no MS → criamos
    if 'DATA' not in t.colnames():
        cd = makearrcoldesc('DATA', complex(0, 0), ndim=2, shape=[1, 1],
                            valuetype='complex')
        t.addcols(maketabdesc(cd))
    t.putcol('UVW',      uvw_m)
    t.putcol('DATA',     data.reshape(n, 1, 1).astype(np.complex64))
    t.putcol('FLAG',     np.zeros((n, 1, 1), dtype=bool))
    t.putcol('WEIGHT',   np.ones((n, 1), dtype=np.float32))
    t.putcol('SIGMA',    np.ones((n, 1), dtype=np.float32))
    t.putcol('ANTENNA1', np.zeros(n, dtype=np.int32))
    t.putcol('ANTENNA2', np.ones(n,  dtype=np.int32))
    t.putcol('TIME',     np.zeros(n, dtype=np.float64))
    t.putcol('FIELD_ID', np.zeros(n, dtype=np.int32))
    t.putcol('DATA_DESC_ID', np.zeros(n, dtype=np.int32))
    t.flush(); t.close()

    # SPECTRAL_WINDOW: 1 canal com a frequência de referência
    spw = table(ms_path + '/SPECTRAL_WINDOW', readonly=False, ack=False)
    if spw.nrows() == 0:
        spw.addrows(1)
    spw.putcell('CHAN_FREQ', 0, np.array([FREQ]))
    spw.putcell('NUM_CHAN', 0, 1)
    spw.flush(); spw.close()

    # Sidecar com as fontes verdadeiras: o C++ lê para validar a recuperação.
    # Mantém make_ms.py e main.cpp em sincronia automática.
    side = os.path.splitext(ms_path)[0] + '.sources.txt'
    with open(side, 'w') as fh:
        fh.write("# px py amp  (nfac=%d)\n" % nfac)
        for (px, py, amp) in SOURCES:
            fh.write("%d %d %g\n" % (px, py, amp))

    print("MS gerado: %s" % ms_path)
    print("  visibilidades: %d  (%d pares hermitianos)" % (n, N_PAIRS))
    print("  freq: %.3f GHz  (wavelength %.4f m)" % (FREQ / 1e9, WL))
    print("  facetas/eixo: %d   fontes: %s" % (nfac, SOURCES))
    print("  sidecar de fontes: %s" % side)

if __name__ == '__main__':
    ms = sys.argv[1] if len(sys.argv) > 1 else 'data/sim.ms'
    nf = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    mi = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    main(ms, nf, mi)

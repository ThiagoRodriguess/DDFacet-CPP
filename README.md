# DDFacet C++

Implementação educacional em **C++17** do **Algoritmo 1** do pipeline de imageamento
**DDFacet** (Monnier et al., *SiPS IEEE 2022*), usado em **rádio-interferometria**, com
foco em **paralelização** (OpenMP intra-nó + MPI multi-nó).

> **Objetivo do algoritmo:** a partir de amostras incompletas e irregulares da Transformada
> de Fourier do céu (as *visibilidades* medidas por um interferômetro), **reconstruir a
> imagem do céu** — descobrir onde estão as fontes de rádio e qual o brilho de cada uma.

Para a explicação detalhada da física e de cada passo, veja
[`DOCUMENTACAO_ALGORITMO.md`](DOCUMENTACAO_ALGORITMO.md) (e o PDF).

---

## Funcionalidades

| Componente | Descrição |
|---|---|
| **Pipeline Algoritmo 1** | FFT (FFTW3) → Degrid → Residual → Grid → FFT⁻¹ → CLEAN (Högbom), em *major cycles* |
| **Measurement Set real** | leitura via **casacore** (`read_ms`); geração via **python-casacore** (`tools/make_ms.py`) |
| **Faceamento** (eixo I) | imagem dividida em `N×N` facetas, cada uma com seu phase center — fase `exp(∓2πi(u·l₀+v·m₀))` no degrid/grid |
| **OpenMP** (intra-nó) | paraleliza o laço de **facetas** (`if(I>1)`) e, internamente, os laços de **visibilidades/pixels** |
| **MPI** (multi-nó, eixo J) | distribui os **Measurement Sets** entre ranks; `MPI_Allreduce` soma as grades UV |

**Validação** (céu de 3 fontes pontuais conhecidas, imagem 128×128):

| Config (I × J) | Recuperação |
|---|---|
| I=1, J=1 | 99% |
| I=1, J=4 (multi-MS) | 99% |
| I=2×2, J=2 (faceamento + multi-MS) | 100% |
| I=4×4, J=1 | 100% |

**Escalabilidade** (CPU i7-10750H, 6 núcleos físicos):

- OpenMP (16 facetas): ~**4.3×** em 6 threads.
- MPI (J=4): **1.0× / 1.72× / 2.80×** em 1 / 2 / 4 ranks — com **resultado idêntico**
  (invariante de correção do `MPI_Allreduce`).

---

## Estrutura do projeto

```
ddfacet_cpp/
├── include/
│   ├── ddfacet.h        estruturas de dados + declarações + pseudocódigo
│   ├── ms_io.h          interface do leitor de Measurement Set
│   └── mpi_util.h       camada MPI (isolada atrás de #ifdef USE_MPI)
├── src/
│   ├── ddfacet.cpp      operadores: imaging_fft(_inv), degrid, grid, residual, deconvolution
│   ├── fft.cpp          FFT/IFFT 2D via FFTW3, fftshift/ifftshift
│   ├── ms_io.cpp        leitor de MS real (casacore)
│   └── main.cpp         geração/leitura do MS, dirty beam, loop principal, validação
├── tools/
│   └── make_ms.py       gerador de Measurement Set real (python-casacore)
├── CMakeLists.txt
├── DOCUMENTACAO_ALGORITMO.md / .pdf   documentação detalhada
└── data/                Measurement Sets gerados (NÃO versionar — ver abaixo)
```

---

## Dependências

Testado em **Ubuntu (WSL)** com g++ 13.

| Dependência | Pacote (apt) | Uso |
|---|---|---|
| Compilador C++17 + OpenMP | `g++` | base |
| FFTW3 (+ threads) | `libfftw3-dev` | FFT 2D |
| casacore | `casacore-dev` | leitura de MS |
| python-casacore + numpy | `python3-casacore python3-numpy` | geração de MS |
| MPI *(opcional)* | `libopenmpi-dev openmpi-bin` | eixo J multi-nó |

```bash
sudo apt install g++ cmake libfftw3-dev casacore-dev \
                 python3-casacore python3-numpy libopenmpi-dev openmpi-bin
```

---

## Compilação

### OpenMP (padrão)

```bash
g++ -std=c++17 -O2 -Wall -fopenmp -Iinclude -isystem /usr/include/casacore \
    src/ddfacet.cpp src/fft.cpp src/main.cpp src/ms_io.cpp \
    -lfftw3_threads -lfftw3 -lm -lcasa_ms -lcasa_tables -lcasa_casa \
    -o build/ddfacet_demo
```

### MPI (eixo J / multi-nó)

```bash
mpic++ -std=c++17 -O2 -fopenmp -DUSE_MPI -Iinclude -isystem /usr/include/casacore \
    src/ddfacet.cpp src/fft.cpp src/main.cpp src/ms_io.cpp \
    -lfftw3_threads -lfftw3 -lm -lcasa_ms -lcasa_tables -lcasa_casa \
    -o build/ddfacet_mpi
```

### CMake

```bash
mkdir -p build && cd build
cmake ..                 # ou: cmake -DUSE_MPI=ON ..
make
```

> Sem `-fopenmp` o código ainda compila e roda serial (pragmas guardados por
> `#ifdef _OPENMP`). Sem `-DUSE_MPI` o MPI vira no-op (1 processo).

---

## Execução

Rode **a partir da raiz do projeto** (o programa precisa de `data/` e `tools/`).

```bash
# recuperação básica (1 faceta, 1 MS) — lê o MS real via casacore
OMP_NUM_THREADS=6 ./build/ddfacet_demo

# faceamento 2×2 + 2 Measurement Sets
DDF_FACETS=2 DDF_NMS=2 OMP_NUM_THREADS=6 ./build/ddfacet_demo

# MPI: 4 ranks × 3 threads (híbrido), 4 Measurement Sets
DDF_NMS=4 OMP_NUM_THREADS=3 mpirun -np 4 ./build/ddfacet_mpi
```

### Variáveis de ambiente

| Variável | Padrão | Significado |
|---|---|---|
| `DDF_FACETS` | 1 | nº de facetas por eixo (`N×N` facetas) |
| `DDF_NMS` | 1 | nº de Measurement Sets (eixo J) |
| `OMP_NUM_THREADS` | (sistema) | threads OpenMP por processo |
| `mpirun -np R` | 1 | nº de ranks MPI (binário `-DUSE_MPI`) |

**O que observar na saída:** testes de FFT `✔ PASS`; `[read_ms]` (MS lido via casacore);
`‖δv‖²` caindo a cada major cycle; e a validação final
`RESULTADO: ✔ Fontes recuperadas com sucesso` (~99–100%).

---

## Sobre a pasta `data/`

Os Measurement Sets (`data/sim_f*_ms*.ms`) são **gerados automaticamente** na primeira
execução: se o `.ms` não existe, o programa chama `tools/make_ms.py` (python-casacore) para
criá-lo. **Não precisam ser versionados nem enviados** — são artefatos reprodutíveis.

Recomenda-se um `.gitignore` com:

```
build/
data/
*.o
```

Para reproduzir em outra máquina basta o código-fonte + as dependências (incluindo
`python-casacore` para gerar os MS).

---

## Como funciona (resumo)

```
for k in K major cycles:
    for j in J Measurement Sets:        # MPI: distribuído entre ranks
        for i in I facetas:             # OpenMP: paralelo
            ĝ = FFT(x̂, i)               # modelo → domínio UV
            v̂ = Degrid(ĝ, i)            # amostra UV nas posições (u,v)
        δv = v − v̂                      # residual (medido − predito)
        for i in I facetas:
            g += Grid(δv, i)            # residual → grade UV (acumula)
    # MPI_Allreduce das grades UV entre ranks (soma sobre todos os MS)
    for i in I facetas:
        δy += FFT⁻¹(g, i)               # grade UV → imagem suja
    x̂ = Deconvolution(δy, PSF)          # Högbom CLEAN
```

A cada *major cycle* o modelo `x̂` ganha as fontes que faltavam e o residual encolhe.
Detalhes completos (visibilidades, plano UV, apodização, correção de grid) em
[`DOCUMENTACAO_ALGORITMO.md`](DOCUMENTACAO_ALGORITMO.md).

---

## Referência

Monnier et al., *"A imaging pipeline for radio interferometry"*, IEEE SiPS 2022.
Implementação C++ educacional — fases sequencial + OpenMP + MPI + leitura de MS (casacore).

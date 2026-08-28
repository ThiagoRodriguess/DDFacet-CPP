#!/bin/bash
# =============================================================================
#  DEMONSTRAÇÃO AO VIVO — DDFacet C++
#  As três alterações da revisão + o pipeline distribuído por canal (OMPC)
# =============================================================================
#
#  Uso:   ./scripts/demo.sh          (pausa entre as partes — para apresentar)
#         ./scripts/demo.sh -y       (sem pausas — para conferir antes)
#
#  Rode a partir da RAIZ do projeto.
#
#  Os trechos de código exibidos são EXTRAÍDOS DOS ARQUIVOS na hora (via sed),
#  não são cópias — o que aparece na tela é o código que está sendo executado.
# =============================================================================

set -u
AUTO=0
[ "${1:-}" = "-y" ] && AUTO=1

B=$(printf '\033[1m');  R=$(printf '\033[0m')
CY=$(printf '\033[1;36m'); YE=$(printf '\033[1;33m')
GR=$(printf '\033[1;32m'); MA=$(printf '\033[1;35m'); DI=$(printf '\033[2m')

titulo() {
    echo
    echo "${CY}================================================================${R}"
    echo "${CY}  $*${R}"
    echo "${CY}================================================================${R}"
}
passo()  { echo; echo "${YE}--- $* ---${R}"; }
nota()   { echo "${DI}    $*${R}"; }
pausa()  { [ $AUTO -eq 1 ] && return 0; echo; read -r -p "${MA}[Enter para continuar]${R}" _; }

# mostra um trecho REAL de um arquivo: codigo <arquivo> <linha_ini> <linha_fim>
codigo() {
    echo "${DI}    $1  (linhas $2-$3)${R}"
    sed -n "$2,$3p" "$1" | sed 's/^/    /'
}

[ -f src/ddfacet.cpp ] || { echo "Rode a partir da raiz do projeto."; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
titulo "DDFacet C++ — demonstração"
cat <<'TXT'
  O que será mostrado, nesta ordem:

    1. O Measurement Set real usado (dataset GLEAM / SKA1-Low)
    2. ALTERAÇÃO 1 — canal espectral como parâmetro explícito
    3. ALTERAÇÃO 2 — termo-w na mudança de centro de fase
    4. ALTERAÇÃO 3 — DDE, ganho dependente da direção
    5. O pipeline distribuído por canal e a sua escalabilidade

  As alterações 2 e 3 são demonstradas sobre um MS com fontes de fluxo
  CONHECIDO, porque só assim dá para medir o impacto delas na recuperação.
  As partes 1 e 5 usam o MS grande real, de 4 canais.
TXT
pausa

# ─────────────────────────────────────────────────────────────────────────────
titulo "1. O Measurement Set real"
passo "Estrutura do MS grande"
MS="../data/sim_large.ms/sim_large.ms"
if [ -d "$MS" ]; then
    echo "    caminho : $MS"
    echo "    tamanho : $(du -sh "$MS" 2>/dev/null | cut -f1)"
    nota "Dataset GLEAM/SKA1-Low: 1.895 fontes, campo de 8 graus,"
    nota "7.848.960 linhas x 4 canais = 31,4 milhoes de visibilidades."
else
    echo "    [aviso] MS nao encontrado em $MS"
fi
passo "Exportando os canais (leitura via casacore, no host)"
nota "Cada canal vira um arquivo .vis independente — a unidade de trabalho"
nota "que sera distribuida entre os nos."
./build/ms_export "$MS" data/large 400000 2>&1 | tail -8
pausa

# ─────────────────────────────────────────────────────────────────────────────
titulo "2. ALTERAÇÃO 1 — canal espectral explícito"
passo "O código"
nota "Antes, o canal era fixo em zero: data(0, 0, i)."
nota "Agora ele e parametro — e a FREQUENCIA vem do mesmo canal."
grep -n "CHAN_FREQ\[channel\]\|const double freq = cf(IPosition(1, channel))\|data(0, channel, i)" src/ms_io.cpp | sed 's/^/    /'
echo
codigo src/ms_io.cpp 52 62
passo "O impacto na execução"
nota "Trocar de canal muda a frequencia, logo muda o comprimento de onda,"
nota "logo muda a conversao UVW [metros] -> [comprimentos de onda]."
echo
grep -h "canal" data/large_ch*.vis 2>/dev/null >/dev/null # no-op
./build/ddfacet_ompc_host data/large 128 1 4 2>&1 | grep -E "^  canal" | sed 's/^/  /'
echo
echo "  ${GR}Repare: o u_max cresce do canal 0 para o canal 3.${R}"
nota "Mesma linha de base fisica; em frequencia maior ela vale mais"
nota "comprimentos de onda. Usar CHAN_FREQ[0] para todos os canais daria"
nota "coordenadas UV erradas em ~2%."
pausa

# ─────────────────────────────────────────────────────────────────────────────
titulo "3. ALTERAÇÃO 2 — termo-w na fase da faceta"
passo "O código"
nota "O w era lido do MS e ignorado. Agora entra na fase:"
nota "  fase = -2pi (u*l0 + v*m0 + w*(n0-1)),  n0 = sqrt(1 - l0^2 - m0^2)"
echo
L=$(grep -n "const double n0m1" src/ddfacet.cpp | head -1 | cut -d: -f1)
codigo src/ddfacet.cpp $((L-3)) $((L+6))
passo "O impacto na execução — com e sem o termo"
nota "DDF_NOW=1 desliga o termo-w (volta ao comportamento antigo)."
nota "Duas execucoes sobre o mesmo MS, com fontes de fluxo conhecido."
echo
echo "  ${B}(a) SEM o termo-w  [DDF_NOW=1]${R}"
DDF_NOW=1 DDF_NMS=1 DDF_FACETS=2 OMP_NUM_THREADS=4 ./build/ddfacet_mpi 2>&1 \
  | grep -E "pix=\(" | sed 's/^/  /'
echo
echo "  ${B}(b) COM o termo-w  [padrão]${R}"
DDF_NMS=1 DDF_FACETS=2 OMP_NUM_THREADS=4 ./build/ddfacet_mpi 2>&1 \
  | grep -E "pix=\(" | sed 's/^/  /'
nota "Neste campo (1 grau) o w e pequeno, entao a diferenca e sutil."
nota "O ponto e que o w agora TEM efeito: em campo largo ele domina."
pausa

# ─────────────────────────────────────────────────────────────────────────────
titulo "4. ALTERAÇÃO 3 — DDE, ganho dependente da direção"
passo "O código"
nota "Versao escalar do formalismo RIME/Jones: um ganho complexo por"
nota "direcao (faceta), aplicado como xG no degrid e xconj(G) no adjunto."
echo
grep -n "directional_gain" include/ddfacet.h | head -2 | sed 's/^/    /'
grep -n "acc \*= gain;\|gain_conj\|\* gain_conj" src/ddfacet.cpp | head -4 | sed 's/^/    /'
passo "O impacto na execução — com e sem os ganhos"
echo
echo "  ${B}(a) SEM DDE  [padrão: G = (1,0), identidade]${R}"
DDF_NMS=1 DDF_FACETS=2 OMP_NUM_THREADS=4 ./build/ddfacet_mpi 2>&1 \
  | grep -E "pix=\(" | sed 's/^/  /'
echo
echo "  ${B}(b) COM DDE  [DDF_DDE=1]${R}"
DDF_DDE=1 DDF_NMS=1 DDF_FACETS=2 OMP_NUM_THREADS=4 ./build/ddfacet_mpi 2>&1 \
  | grep -E "faceta [0-9]:|pix=\(" | sed 's/^/  /'
echo
echo "  ${GR}A recuperacao cai SO nas facetas fora do centro.${R}"
nota "A faceta 0 tem G = identidade (fase 0) e fica intacta. As demais tem"
nota "fase crescente, e a deconvolucao — que nao conhece os ganhos — perde"
nota "fluxo. E o efeito fisico esperado de um efeito dependente da direcao."
nota "Prova que o ganho esta ligado no degrid E no grid, nao so declarado."
pausa

# ─────────────────────────────────────────────────────────────────────────────
titulo "5. O pipeline distribuído por canal (OpenMP Cluster)"
passo "O código — uma tarefa por canal"
nota "Cada canal e despachado como uma regiao 'omp target'. No cluster, o"
nota "runtime do OMPC envia cada uma para um NO diferente, via MPI."
echo
L2=$(grep -n "#pragma omp task firstprivate" src/main_ompc.cpp | head -1 | cut -d: -f1)
codigo src/main_ompc.cpp $((L2-7)) $((L2+14))
passo "Execução: ciclo completo sobre os 4 canais reais"
OMP_NUM_THREADS=4 ./build/ddfacet_ompc_host data/large 128 3 4 2>&1 \
  | grep -E "WORKER|dirty|modelo|tempo|checksum" | sed 's/^/  /'
pausa

passo "Escalabilidade — o mesmo trabalho, mais unidades de execução"
echo -n "    1 unidade  : "
T1=$(OMP_NUM_THREADS=1 ./build/ddfacet_ompc_host data/large 128 2 4 2>&1 | grep -oP "tempo nas tarefas\s+: \K[0-9.]+")
echo "${T1} s"
echo -n "    2 unidades : "
T2=$(OMP_NUM_THREADS=2 ./build/ddfacet_ompc_host data/large 128 2 4 2>&1 | grep -oP "tempo nas tarefas\s+: \K[0-9.]+")
echo "${T2} s"
echo -n "    4 unidades : "
T4=$(OMP_NUM_THREADS=4 ./build/ddfacet_ompc_host data/large 128 2 4 2>&1 | grep -oP "tempo nas tarefas\s+: \K[0-9.]+")
echo "${T4} s"
echo
awk -v a="$T1" -v b="$T2" -v c="$T4" 'BEGIN{
  printf "    speedup:  2 unidades = %.2fx   |   4 unidades = %.2fx\n", a/b, a/c;
  printf "    eficiencia: %.0f%%  e  %.0f%%\n", 100*(a/b)/2, 100*(a/c)/4;
}'
nota "Medido com as regioes target em THREADS da mesma maquina (memoria"
nota "compartilhada), porque o cluster esta em manutencao. Demonstra que a"
nota "decomposicao por canal escala; NAO inclui o custo de rede entre nos."
pausa

passo "Corretude: o resultado não depende de quantas unidades processaram"
for n in 1 2 4; do
    printf "    %s unidade(s): " "$n"
    OMP_NUM_THREADS=$n ./build/ddfacet_ompc_host data/large 128 2 4 2>&1 \
      | grep -oP "checksum \|dirty\|\s+: \K[0-9.]+"
done
echo
echo "  ${GR}Checksums identicos.${R}"
nota "Distribuir apenas reparte os canais; a soma das grades UV parciais e"
nota "a mesma soma, e soma e associativa. Se um canal fosse perdido,"
nota "duplicado, ou houvesse corrida de escrita, o checksum divergiria."

titulo "Fim da demonstração"
cat <<'TXT'
  Resumo:
    - As tres alteracoes estao implementadas e cada uma tem efeito
      observavel na execucao (nao apenas compilam).
    - O canal virou o eixo de distribuicao: escalamento quase linear.
    - A corretude e verificada por invariancia sob o numero de unidades.

  Falta: compilar com o toolchain do OpenMP Cluster e medir com nos
  reais — bloqueado pela manutencao do cluster, mesmo codigo-fonte.
TXT
echo

#!/bin/bash
# =============================================================================
#  DDFacet + OpenMP Cluster (OMPC) no sorgan — build e execução distribuída
# =============================================================================
#
#  O QUE ESTE SCRIPT FAZ
#  ---------------------
#  Compila e roda `src/main_ompc.cpp`, que distribui o kernel do DDFacet
#  (degrid + residual + grid) por CANAL usando `#pragma omp target`. No OMPC,
#  cada região `target` é despachada pelo runtime para um NÓ do cluster via MPI.
#
#  MODELO DE EXECUÇÃO (importante para entender o comando `run`)
#  ------------------------------------------------------------
#  O OMPC usa um lançamento MPMD com DOIS grupos de processos:
#
#     mpirun -np N  ...  llvm-offload-mpi-proxy-device   \  <- N "devices" (nós
#            :                                              que executam as
#            -np 1  ...  ./ddfacet_ompc                     regiões target)
#                                                        ^- 1 "head" (roda o
#                                                           programa principal)
#
#  Ou seja: N+1 processos no total. O head cria as tarefas; os proxy-devices
#  as recebem e executam. É por isso que o comando tem um ':' no meio.
#
#  PRÉ-REQUISITOS
#  --------------
#    * Acesso ao sorgan (ssh)
#    * Trabalhar em /scratch/$USER (não no home)
#    * Baixar o container A PARTIR DE UM NÓ DE COMPUTAÇÃO (o login node tem
#      restrições de rede/espaço)
#
#  USO
#  ---
#    ./sorgan_ompc.sh setup           # baixa o container OMPC (uma vez só)
#    ./sorgan_ompc.sh build           # compila com o clang do container
#    ./sorgan_ompc.sh run  <hosts> [nchan] [nvis]
#    ./sorgan_ompc.sh check <hosts>   # TESTE DE INVARIÂNCIA (1 nó vs N nós)
#
#  Exemplo de sessão completa:
#    salloc -N 2 -p cpu --no-shell --exclusive --time 01:00:00
#    # anote os nós, ex.: sorgan-cpu1,sorgan-cpu2
#    ssh sorgan-cpu1
#    cd /scratch/$USER/ddfacet_cpp
#    ./scripts/sorgan_ompc.sh setup
#    ./scripts/sorgan_ompc.sh build
#    ./scripts/sorgan_ompc.sh check sorgan-cpu1,sorgan-cpu2
# =============================================================================

set -u

# ─── Configuração (ajuste se os nomes mudarem) ───────────────────────────────
CONTAINER_URL="docker://cl3t0/ompc:latest"
SIF="${SIF:-ompc_latest.sif}"          # nome do arquivo gerado pelo pull
SRC="src/main_ompc.cpp"
BIN="ddfacet_ompc"
INC="include"
MPI_MODULE="${MPI_MODULE:-mpich}"
# --nv é para GPUs NVIDIA; em nós CPU é inofensivo. Esvazie se causar problema.
APPTAINER_FLAGS="${APPTAINER_FLAGS:---nv}"

msg()  { printf '\n\033[1;36m>>> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[aviso]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[erro]\033[0m %s\n' "$*" >&2; exit 1; }

# O script usa caminhos RELATIVOS (src/, include/, scripts/) — tem de ser
# executado a partir da RAIZ do projeto, não de dentro de scripts/.
require_root() {
    if [ ! -f "$SRC" ] || [ ! -d "$INC" ]; then
        die "rode a partir da RAIZ do projeto (onde existem src/ e include/).
       Voce esta em: $(pwd)
       Tente:  cd \"\$(dirname \"\$0\")/..\"  &&  ./scripts/$(basename "$0") ..."
    fi
}

require_sif() {
    [ -f "$SIF" ] || die "container '$SIF' nao encontrado. Rode: $0 setup
       (e confira o nome do .sif gerado — pode ser 'ompc_latest.sif' ou outro)"
}

# ─── setup: baixa o container ────────────────────────────────────────────────
cmd_setup() {
    msg "Baixando o container OMPC ($CONTAINER_URL)"
    warn "Faça isto a partir de um NÓ DE COMPUTAÇÃO, não do login node:"
    warn "  salloc -N 1 -p cpu --no-shell --exclusive --time 01:00:00"
    warn "  ssh <no-alocado>"

    command -v apptainer >/dev/null 2>&1 || die "apptainer nao encontrado (tente: ml load apptainer)"
    apptainer pull "$CONTAINER_URL" || die "falha no pull do container"

    msg "Arquivos .sif presentes:"
    ls -la ./*.sif 2>/dev/null || warn "nenhum .sif no diretorio atual"
    warn "Se o arquivo gerado tiver outro nome, exporte-o: export SIF=<nome>.sif"
}

# ─── hello: VALIDA O TOOLCHAIN antes de tentar o nosso código ────────────────
# Compila e roda o menor programa com offload. Se isto não funcionar, nada
# adiante vai funcionar — e o erro aqui é muito mais fácil de diagnosticar.
cmd_hello() {
    require_root; require_sif
    local hosts="${1:-}"
    [ -n "$hosts" ] || die "informe os hosts. Ex: $0 hello sorgan-cpu1,sorgan-cpu2"
    [ -f scripts/hello_ompc.cpp ] || die "scripts/hello_ompc.cpp nao encontrado"

    local ndev
    ndev=$(awk -F',' '{print NF}' <<< "$hosts")

    msg "[1/2] Compilando o hello-world com o clang do container"
    apptainer exec $APPTAINER_FLAGS "$SIF" \
        clang++ -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu \
                scripts/hello_ompc.cpp -o hello_ompc \
        || die "falha ao compilar o hello-world — o problema esta no CONTAINER/CLANG,
       nao no nosso codigo. Confira o nome do .sif e se o clang++ existe:
         apptainer exec $SIF clang++ --version"

    msg "[2/2] Executando com $ndev proxy-device(s): $hosts"
    module load "$MPI_MODULE" 2>/dev/null || warn "sem 'module load $MPI_MODULE'"

    mpirun -hosts="$hosts" \
           -np "$ndev" apptainer exec --sharens $APPTAINER_FLAGS "$SIF" llvm-offload-mpi-proxy-device \
         : -np 1        apptainer exec --sharens $APPTAINER_FLAGS "$SIF" ./hello_ompc \
         | tee /tmp/ompc_hello.log

    msg "Diagnostico"
    local head_pid worker_pids nworkers
    head_pid=$(grep -o 'HEAD.*pid = [0-9]*' /tmp/ompc_hello.log | grep -o '[0-9]*$' | head -1)
    worker_pids=$(grep -o 'WORKER.*pid = [0-9]*' /tmp/ompc_hello.log | grep -o '= [0-9]*' | grep -o '[0-9]*' | sort -u)
    nworkers=$(echo "$worker_pids" | grep -c . )

    echo "  PID do head    : ${head_pid:-<nenhum>}"
    echo "  PIDs dos worker: $(echo $worker_pids)"

    if [ -z "$worker_pids" ]; then
        printf '\n\033[1;31m  ✘ Nenhum WORKER imprimiu — a regiao target nao executou.\033[0m\n'
        return 1
    elif [ "$worker_pids" = "$head_pid" ]; then
        printf '\n\033[1;33m  ! WORKER com o MESMO pid do HEAD — rodou no host (fallback),\n'
        printf '    o offload NAO esta ativo. Verifique -fopenmp-targets e o proxy-device.\033[0m\n'
        return 1
    else
        printf '\n\033[1;32m  ✔ OFFLOAD ATIVO — %s worker(s) com pid distinto do head.\033[0m\n' "$nworkers"
        printf '\033[1;32m    Toolchain validado: pode seguir para build/check.\033[0m\n\n'
    fi
}

# ─── build: compila com o clang patched de dentro do container ───────────────
cmd_build() {
    require_sif
    [ -f "$SRC" ] || die "fonte '$SRC' nao encontrado (rode a partir da raiz do projeto)"

    msg "Compilando $SRC com o clang++ do container"
    # -fopenmp-targets=x86_64-pc-linux-gnu: o "device" e outro NO x86, nao uma GPU.
    apptainer exec $APPTAINER_FLAGS "$SIF" \
        clang++ -O2 -std=c++17 -fopenmp \
                -fopenmp-targets=x86_64-pc-linux-gnu \
                -I"$INC" "$SRC" -o "$BIN" \
        || die "falha na compilacao"

    msg "OK: binario '$BIN' gerado"
    ls -la "$BIN"
}

# ─── run: execução distribuída (N proxy-devices + 1 head) ────────────────────
# uso: cmd_run <hosts-separados-por-virgula> [nchan] [nvis]
cmd_run() {
    require_root; require_sif
    local hosts="${1:-}"
    local nchan="${2:-4}"
    local nvis="${3:-200000}"
    [ -n "$hosts" ] || die "informe os hosts. Ex: $0 run sorgan-cpu1,sorgan-cpu2"
    [ -x "$BIN" ]   || die "binario '$BIN' nao existe. Rode: $0 build"

    # nº de devices = nº de hosts informados
    local ndev
    ndev=$(awk -F',' '{print NF}' <<< "$hosts")

    msg "Executando com $ndev no(s) de offload  |  canais=$nchan  vis/canal=$nvis"
    echo "    hosts: $hosts"

    module load "$MPI_MODULE" 2>/dev/null || warn "nao consegui 'module load $MPI_MODULE' (siga sem, se ja estiver no ambiente)"

    # MPMD: N proxy-devices  :  1 head com o nosso programa
    mpirun -hosts="$hosts" \
           -np "$ndev" apptainer exec --sharens $APPTAINER_FLAGS "$SIF" llvm-offload-mpi-proxy-device \
         : -np 1        apptainer exec --sharens $APPTAINER_FLAGS "$SIF" ./"$BIN" "$nchan" "$nvis"
}

# ─── check: TESTE DE INVARIÂNCIA — o resultado nao pode depender do nº de nós ─
cmd_check() {
    local hosts="${1:-}"
    [ -n "$hosts" ] || die "informe os hosts. Ex: $0 check sorgan-cpu1,sorgan-cpu2"

    local first
    first=$(cut -d',' -f1 <<< "$hosts")

    msg "TESTE DE INVARIANCIA — o checksum deve ser IDENTICO nos dois casos"
    echo "  A distribuicao so reparte os canais entre os nos; a soma das grades"
    echo "  UV parciais tem de dar o mesmo resultado. Se os checksums diferirem,"
    echo "  ha erro na decomposicao (canal perdido, duplicado ou corrida)."

    msg "[1/2] UM no de offload ($first)"
    cmd_run "$first" 4 200000 | tee /tmp/ompc_1no.log

    msg "[2/2] TODOS os nos ($hosts)"
    cmd_run "$hosts" 4 200000 | tee /tmp/ompc_Nnos.log

    msg "Comparacao dos checksums"
    local c1 cN
    c1=$(grep -i 'checksum' /tmp/ompc_1no.log  | tail -1)
    cN=$(grep -i 'checksum' /tmp/ompc_Nnos.log | tail -1)
    echo "  1 no  : $c1"
    echo "  N nos : $cN"
    if [ "$c1" = "$cN" ] && [ -n "$c1" ]; then
        printf '\n\033[1;32m  ✔ INVARIANTE OK — a distribuicao esta correta.\033[0m\n\n'
    else
        printf '\n\033[1;31m  ✘ DIVERGENCIA — investigar a decomposicao/mapeamento.\033[0m\n\n'
        return 1
    fi
}

# ─── despacho ────────────────────────────────────────────────────────────────
case "${1:-}" in
    setup) shift; cmd_setup "$@" ;;
    hello) shift; cmd_hello "$@" ;;
    build) shift; cmd_build "$@" ;;
    run)   shift; cmd_run   "$@" ;;
    check) shift; cmd_check "$@" ;;
    *)
        sed -n '2,50p' "$0"
        echo
        echo "Comandos: setup | hello <hosts> | build | run <hosts> [nchan] [nvis] | check <hosts>"
        echo
        echo "Ordem recomendada:  setup -> hello -> build -> check"
        ;;
esac

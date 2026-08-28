/**
 * @file hello_ompc.cpp
 * @brief Menor programa possível que prova que o offload do OMPC funciona.
 *
 * (Do tutorial da Ophélie Renaud — LSC/Unicamp.)
 *
 * O QUE OBSERVAR NA SAÍDA
 * -----------------------
 * O programa imprime o PID do processo "head" e o PID de quem executou a região
 * `omp target`. O sinal de que o OMPC está REALMENTE distribuindo é:
 *
 *     [ HEAD ] pid = 12345 ...
 *     [WORKER] pid = 67890 (head pid 12345 ...)   <- PID DIFERENTE
 *
 * Se o PID do WORKER for IGUAL ao do HEAD, a região target rodou no próprio
 * host (fallback), ou seja: o offload NÃO está ativo — o binário foi compilado
 * sem -fopenmp-targets, ou o proxy-device não subiu.
 *
 * Com N proxy-devices você deve ver N PIDs distintos entre os workers.
 */
#include <cstdio>
#include <unistd.h>
#include <omp.h>

int main(int argc, char **argv) {
    (void)argc; (void)argv;

    int pid = getpid();
    printf("[ HEAD ] pid = %d @ %p\n", pid, (void*)&pid);

    #pragma omp parallel
    #pragma omp single
    {
        #pragma omp target nowait depend(in: pid)
        {
            printf("[WORKER] pid = %d (head pid %d @ %p)\n",
                   getpid(), pid, (void*)&pid);
        }
    }

    return 0;
}

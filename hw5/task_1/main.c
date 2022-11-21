#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_WAIT_ITERS 100000
#define FIX_UNUSED(X) (void)(X)
#define CHECK_RET(RET, MSG)                                \
    {                                                      \
        int local_ret = (RET);                             \
        if (local_ret) {                                   \
            printf("Error: %s: ret = %d\n", MSG, local_ret); \
            return -1;                                     \
        }                                                  \
    }

int generate_name(int prank, int psize) {
    FIX_UNUSED(psize);
    unsigned int seed = (prank + 1) * (unsigned int) time(NULL);
    return (int) rand_r(&seed);
}

int choose_next(int prank, int psize, int prev_ranks_size, const int *prev_ranks) {
    unsigned int seed = (prank + 1) * (unsigned int) time(NULL);
    if (prev_ranks_size >= psize)
        return -1;
    while (1) {
        int next_rank = (int) rand_r(&seed) % psize;
        int free_rank = 1;
        for (int i = 0; i < prev_ranks_size; i++) {
            if (next_rank == prev_ranks[i]) {
                free_rank = 0;
                break;
            }
        }
        if (free_rank)
            return next_rank;
    }
    return -1;
}

int send_to_next(int prank, int psize, int prev_ranks_size, int* prev_ranks, int* prev_names) {
    if (prev_ranks_size == psize) {
        printf("Last rank in a game: %d\n", prank);
        printf("Game sequence: ");
        for (int i = 0; i < psize; i++) {
            printf("%d -> ", prev_ranks[i]);
        }
        printf("\n");
        printf("Names: ");
        for (int i = 0; i < psize; i++) {
            printf("%d -> ", prev_names[i]);
        }
        printf("\n");
    }
    else {
        int next_rank = choose_next(prank, psize, prev_ranks_size, prev_ranks);
        MPI_Request request;
        CHECK_RET(MPI_Isend(&prev_ranks_size, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD, &request),
                  "MPI_Send: prev_ranks_size");
        CHECK_RET(MPI_Ssend(prev_ranks, prev_ranks_size, MPI_INT, next_rank, 0, MPI_COMM_WORLD),
                  "MPI_Send: prev_ranks");
        CHECK_RET(MPI_Ssend(prev_names, prev_ranks_size, MPI_INT, next_rank, 0, MPI_COMM_WORLD),
                  "MPI_Send: prev_names");
    }
    return 0;
}

int start_communication(int prank, int psize, int *ret_src, int *ret_prev_ranks_size) {
    int prev_ranks_size = -1;
    int src_prank = -1;
    MPI_Request *requests = (MPI_Request *) malloc(psize * sizeof(MPI_Request));
    for (src_prank = 0; src_prank < psize; src_prank++) {
        if (src_prank == prank)
            continue;
        CHECK_RET(
                MPI_Irecv(&prev_ranks_size, 1, MPI_INT, src_prank, 0, MPI_COMM_WORLD, &requests[src_prank]),
                "MPI_Recv: prev_ranks_size");
    }
    for (int iter = 0; iter < MAX_WAIT_ITERS; iter++) {
        int flag = 0;
        for (src_prank = 0; src_prank < psize; src_prank++) {
            if (src_prank == prank)
                continue;
            CHECK_RET(MPI_Test(&requests[src_prank], &flag, MPI_STATUS_IGNORE), "MPI_Test");
            if (flag) {
                CHECK_RET(MPI_Wait(&requests[src_prank], MPI_STATUS_IGNORE), "MPI_Wait");
                break;
            }
        }
        if (flag) {
            break;
        }
    }
    free(requests);
    *ret_src = src_prank;
    *ret_prev_ranks_size = prev_ranks_size;
    return 0;
}

int play_game(int prank, int psize) {
    if (prank == 0) {
        // start game
        int prev_ranks_size = 1;
        int name = generate_name(prank, psize);
        send_to_next(prank, psize, prev_ranks_size, &prank, &name);
    } else {
        // waiting for communication
        int src_prank = -1;
        int prev_ranks_size = -1;
        start_communication(prank, psize, &src_prank, &prev_ranks_size);
        if (src_prank == psize || src_prank < 0) {
            return -1;
        }
        // get prev ranks from src
        int *prev_ranks = (int *) malloc((prev_ranks_size + 1) * sizeof(int));
        int *prev_names = (int *) malloc((prev_ranks_size + 1) * sizeof(int));
        CHECK_RET(
                MPI_Recv(prev_ranks, prev_ranks_size, MPI_INT, src_prank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                "MPI_Recv: prev_ranks");
        CHECK_RET(
                MPI_Recv(prev_names, prev_ranks_size, MPI_INT, src_prank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                "MPI_Recv: prev_names");
        prev_ranks[prev_ranks_size] = prank;
        prev_names[prev_ranks_size] = generate_name(prank, psize);
        prev_ranks_size++;
        // continue game
        send_to_next(prank, psize, prev_ranks_size, prev_ranks, prev_names);
        free(prev_ranks);
        free(prev_names);
    }
    return 0;
}

int main(int argc, char **argv) {
    int prank, psize;
    CHECK_RET(MPI_Init(&argc, &argv), "MPI_Init");
    CHECK_RET(MPI_Comm_rank(MPI_COMM_WORLD, &prank), "MPI_Comm_rank");
    CHECK_RET(MPI_Comm_size(MPI_COMM_WORLD, &psize), "MPI_Comm_size");
    CHECK_RET(play_game(prank, psize), "play_game");
    CHECK_RET(MPI_Finalize(), "MPI_Finalize");
    return 0;
}

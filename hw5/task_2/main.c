#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define DEBUG 1
#define VERBOSE 1
#define CYCLIC_WORLD 1
#define GHOST_OFFSET 1

#define WORLD_SIZE 100
#define MAX_ITERS 20
#define RULE 90

#define FIX_UNUSED(X) (void)(X)
#define CHECK_RET(RET, MSG)                                                    \
  if (RET) {                                                                   \
    printf("Error: %s: ret = %d", MSG, RET);                                   \
    return -1;                                                                 \
  }

// [ G ] [ . . . ] [ G ]
int get_N(int prank, int psize) {
    FIX_UNUSED(prank);
    assert(WORLD_SIZE > psize);
    return WORLD_SIZE / psize +
           2 * GHOST_OFFSET +
           (prank < WORLD_SIZE % psize);
}

int prev_rank(int prank, int psize) {
    int prev = prank - 1;
    if (CYCLIC_WORLD)
        return (prev + psize) % psize;
    else
        return prev;
}

int next_rank(int prank, int psize) {
    int next = prank + 1;
    if (CYCLIC_WORLD)
        return next % psize;
    else
        return next;
}

int begin(int N) {
    FIX_UNUSED(N);
    return GHOST_OFFSET;
}

int end(int N) {
    assert(N >= 3 * GHOST_OFFSET);
    return N - GHOST_OFFSET;
}

int count(int N) {
    return end(N) - begin(N);
}

void init_state(int prank, int psize, unsigned int *state) {
    int N = get_N(prank, psize);
    unsigned int seed = (prank + 1) * (unsigned int) time(NULL);
    for (int i = 0; i < begin(N); i++)
        state[i] = 0;
    for (int i = begin(N); i < end(N); i++)
        state[i] = (unsigned int) rand_r(&seed) % 2;
    for (int i = end(N); i < N; i++)
        state[i] = 0;
}

int sent_recv_ghost_cells(int prank, int psize, int to_next, unsigned int *state) {
    int N = get_N(prank, psize);
    int next = next_rank(prank, psize);
    int prev = prev_rank(prank, psize);
    int dst_prank = to_next ? next : prev;
    int src_prank = to_next ? prev : next;
    int src_offset = to_next ? end(N) - GHOST_OFFSET : begin(N);
    int dst_offset = to_next ? 0 : end(N);
    // to_next
    // src: [ G ] [  . . . ] [ G ]
    // dst:            [ G ] [  . . . ] [ G ]
    // !to_next
    // src:            [ G ] [  . . . ] [ G ]
    // dst: [ G ] [  . . . ] [ G ]
    if (dst_prank >= 0 && dst_prank < psize)
        CHECK_RET(
                MPI_Send(state + src_offset, GHOST_OFFSET, MPI_UNSIGNED, dst_prank, 0, MPI_COMM_WORLD),
                "MPI_Send");
    if (src_prank >= 0 && src_prank < psize)
        CHECK_RET(
                MPI_Recv(state + dst_offset, GHOST_OFFSET, MPI_UNSIGNED, src_prank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                "MPI_Recv");
    return 0;
}

int make_step(int ghost_lo, int ghost_hi, int lo, int hi, unsigned int *state) {
    assert(RULE >= 0 && RULE < 256);
    int converged = 1;
    unsigned int prev = ghost_lo;
    for (int i = lo; i < hi; i++) {
        unsigned int curr = state[i];
        unsigned int next = i + 1 < hi ? state[i + 1] : ghost_hi;
        unsigned int val = (next) ^ (curr << 1) ^ (prev << 2);
        state[i] = 0;
        for (unsigned int pos = 0; pos < 8; pos++) {
            if (RULE & (1 << pos))
                state[i] |= (val == pos);
        }
        if (DEBUG) {
            if (RULE == 110)
                assert(state[i] == (val == 1 || val == 2 || val == 3 || val == 5 || val == 6));
            else if (RULE == 90)
                assert(state[i] == (val == 1 || val == 3 || val == 4 || val == 6));
            else if (RULE == 30)
                assert(state[i] == (val == 1 || val == 2 || val == 3 || val == 4));
        }
        if (curr != state[i])
            converged = 0;
        prev = curr;
    }
    return converged;
}

int gather_states(int prank, int psize, const unsigned int *state, unsigned int *full_state) {
    int N = get_N(prank, psize);
    if (prank == 0) {
        for (int i = begin(N), j = 0; i < end(N); i++, j++) {
            full_state[j] = state[i];
        }
        for (int src = 1, offset = count(N); src < psize; src++) {
            int src_N = get_N(src, psize);
            int src_count = count(src_N);
            CHECK_RET(
                    MPI_Recv(full_state + offset, src_count, MPI_UNSIGNED, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE),
                    "MPI_Recv");
            offset += src_count;
        }
    } else {
        int dst = 0;
        CHECK_RET(
                MPI_Send(state + begin(N), count(N), MPI_UNSIGNED, dst, 0, MPI_COMM_WORLD),
                "MPI_Send");
    }
    return 0;
}

int run_wolfram_automaton(int prank, int psize) {
    int N = get_N(prank, psize);
    unsigned int *state = NULL, *full_state = NULL, *prev_full_state = NULL;
    state = (unsigned int *) malloc(N * sizeof(unsigned int));
    if (prank == 0 && DEBUG) {
        full_state = (unsigned int *) malloc(WORLD_SIZE * sizeof(unsigned int));
        prev_full_state = (unsigned int *) malloc(WORLD_SIZE * sizeof(unsigned int));
    }

    // initialize cells
    init_state(prank, psize, state);
    if (DEBUG) {
        gather_states(prank, psize, state, prev_full_state);
        if (prank == 0 && VERBOSE) {
            for (int i = 0; i < WORLD_SIZE; i++) {
                printf("%u", prev_full_state[i]);
            }
            printf("\n");
        }
    }

    int iter = 0;
    int converged = 0;
    while (!converged && iter < MAX_ITERS) {
        // share ghost cells
        if (psize > 1) {
            CHECK_RET(sent_recv_ghost_cells(prank, psize, 1 /*to next*/, state), "sent/recv ghost cells to next");
            CHECK_RET(sent_recv_ghost_cells(prank, psize, 0 /*to next*/, state), "sent/recv ghost cells to prev");
        }
        else if (CYCLIC_WORLD) {
            for (int i = 0; i < begin(N); i++)
                state[i] = state[end(N) - GHOST_OFFSET + i];
            for (int i = end(N); i < N; i++)
                state[i] = state[GHOST_OFFSET + i - end(N)];
        }

        // calculate next state
        int ghost_lo = state[begin(N) - 1];
        int ghost_hi = state[end(N)];
        converged = make_step(ghost_lo, ghost_hi, begin(N), end(N), state);
        CHECK_RET(MPI_Allreduce(
                          &converged,
                          &converged,
                          1,
                          MPI_INT,
                          MPI_LAND,
                          MPI_COMM_WORLD),
                  "MPI_Allreduce");
        iter++;

        // check result with sequential version
        if (DEBUG) {
            gather_states(prank, psize, state, full_state);
            if (prank == 0) {
                if (VERBOSE) {
                    for (int i = 0; i < WORLD_SIZE; i++) {
                        printf("%u", full_state[i]);
                    }
                    printf("\n");
                }
                ghost_lo = CYCLIC_WORLD ? prev_full_state[WORLD_SIZE-1] : 0;
                ghost_hi = CYCLIC_WORLD ? prev_full_state[0] : 0;
                assert(converged == make_step(ghost_lo, ghost_hi, 0, WORLD_SIZE, prev_full_state));
                for (int i = 0; i < WORLD_SIZE; i++) {
                    assert(full_state[i] == prev_full_state[i]);
                }
                memcpy(prev_full_state, full_state, WORLD_SIZE * sizeof(unsigned int));
            }
        }
    }

    free(state);
    if (prank == 0 && DEBUG) {
        free(full_state);
        free(prev_full_state);
    }
    return 0;
}

int main(int argc, char **argv) {
    int prank, psize;
    CHECK_RET(MPI_Init(&argc, &argv), "MPI_Init");
    CHECK_RET(MPI_Comm_rank(MPI_COMM_WORLD, &prank), "MPI_Comm_rank");
    CHECK_RET(MPI_Comm_size(MPI_COMM_WORLD, &psize), "MPI_Comm_size");


    double time_elapsed = MPI_Wtime();
    run_wolfram_automaton(prank, psize);
    time_elapsed = MPI_Wtime() - time_elapsed;
    if (prank == 0)
        printf("Time: %lf\n", time_elapsed);

    CHECK_RET(MPI_Finalize(), "MPI_Finalize");
    return 0;
}

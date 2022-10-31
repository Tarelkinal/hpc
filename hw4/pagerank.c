#include <omp.h>
#include <stdio.h>
#include <ctime>
#include <stdlib.h>
#include <math.h>

const double EPS = 0.0001;

void seedThreads(const size_t nThreads, unsigned int* seeds) {
    int my_thread_id;
    unsigned int seed;
    #pragma omp parallel private (seed, my_thread_id)
    {
        my_thread_id = omp_get_thread_num();
        unsigned int seed = (unsigned) time(NULL);
        seeds[my_thread_id] = (seed & 0xFFFFFFF0) | (my_thread_id + 1);
    }
    return;
}


void print_matrix(const size_t N, const double *M)
{
    for (size_t i = 0; i < N; ++i)
    {    
        for (size_t j = 0; j < N; ++j)
            printf("%f ", M[i * N + j]); 
        printf("\n"); 
    }
    return;
}


void print_vector(const size_t N, const double *V)
{
    for (size_t i = 0; i < N; ++i)
        printf("%f ", V[i]); 
    printf("\n"); 
    return;
}


void norm_cols_matrix(const size_t N, double *M)
{
    double sum[N];
#pragma omp parallel for 
    for (size_t i = 0; i < N; ++i)   
        sum[i] = 0;

#pragma omp parallel for 
    for (size_t i = 0; i < N; ++i)   
        for (size_t j = 0; j < N; ++j)
                sum[i] += M[j * N + i];
#pragma omp parallel for 
    for (size_t i = 0; i < N; ++i)   
        for (size_t j = 0; j < N; ++j)
            if (abs(sum[j]) > EPS)
                M[i * N + j] /= sum[j];
    return;
}


double norm2(const size_t N, const double *V)
{
    double norm = 0;
#pragma omp parallel for reduction(+: norm)
    for (size_t i = 0; i < N; ++i)   
        norm += V[i] * V[i];
    return sqrt(norm);
}


double norm2_vector(const size_t N, double *V)
{
    const double norm = norm2(N, &V[0]);

    if (abs(norm) > EPS)
    {
#pragma omp parallel for 
    for (size_t i = 0; i < N; ++i)   
        V[i] /= norm;
    }
    return norm;
}


void mult_matrix_on_vector(const size_t N, const double *M, const double *V, double *res)
{
    double tmp[N];
#pragma omp parallel for 
    for (size_t i = 0; i < N; ++i)   
        tmp[i] = 0;
#pragma omp parallel for 
    for (size_t i = 0; i < N; ++i)   
        for(size_t k = 0; k < N; ++k)
            tmp[i] += M[i * N + k] * V[k];
#pragma omp parallel for 
    for (size_t i = 0; i < N; ++i)   
        res[i] = tmp[i];
    return;
}


void pagerank(const size_t N, double *M, double *x)
{
    // initial approximation
#pragma omp parallel for 
    for (size_t i = 0; i < N; ++i)   
        x[i] = 1;

    double norm_cur, norm_last;
    norm_last = norm2_vector(N, &x[0]);

    printf("pagerank iterations\nnorm = %f\n", norm_last);

    // norm graph columns
    norm_cols_matrix(N, M);
    
    // 1st iteration
    mult_matrix_on_vector(N, &M[0], &x[0], &x[0]);
    norm_cur = norm2_vector(N, &x[0]);

    // iteration process
    for (; abs(norm_cur - norm_last) >= EPS;)  
    {
        printf("norm = %f\n", norm_cur);
        norm_last = norm_cur;
        mult_matrix_on_vector(N, &M[0], &x[0], &x[0]);
        norm_cur = norm2_vector(N, &x[0]);
    }   
    printf("norm = %f\n", norm_cur);

    return;
}

int main(int argc, char* argv[])
{
    const size_t nThreads = (argc > 3) ? atoi(argv[3]) : 4;

    srand(time(NULL));
    unsigned int seeds[nThreads];
    omp_set_num_threads(nThreads);
    seedThreads(nThreads, seeds);

    size_t tid, seed;
    const size_t N = (argc > 1) ? atoi(argv[1]) : 5;
    const size_t MAX_LINK_COUNT = (argc > 2) ? atoi(argv[2]) : 1;
    double M[N * N];
    double ranks[N];
    double naiveranks[N];

#pragma omp parallel private(tid, seed)
    {
        tid = omp_get_thread_num();
        seed = seeds[tid];
        srand(seed);
// generate graph
#pragma omp for 
        for (size_t i = 0; i < N; ++i)
        {    
            for (size_t j = 0; j < N; ++j)
                if (i != j)
                    M[i * N + j] = rand() % (MAX_LINK_COUNT + 1);
                else
                    M[i * N + j] = 0;
        }
    }

// print graph
    printf("\ngraph\n");
    print_matrix(N, M);
    printf("\n");

//  calc naive ranks
#pragma omp parallel for 
    for (size_t i = 0; i < N; ++i)
    {    
        naiveranks[i] = 0;
        for (size_t j = 0; j < N; ++j)
            naiveranks[i] += M[i * N + j];      
    }

// pagerank
    pagerank(N, &M[0], &ranks[0]);
    printf("\npagerank - ranks\n");
    print_vector(N, &ranks[0]);

// naive ranks
    printf("\nnaive ranking - inner links count\n");
    print_vector(N, &naiveranks[0]);
    printf("\n");

    return 0;
}

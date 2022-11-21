# HPC server and to Linux-based cluster. Pagerank algorithm  

## Pagerank, matrices and graphs

1. Generate random graphs and test A^m counter of number of paths. Evaluate it within log2(m) matmul-s.  
2. Generate small random or select some graph and implement pagerank procedure. Compare with naive ranking.  
• Tip: Moscow subway graph is a nice example (but rather tiny).  
3. Try all tricks for evaluation of performance: optimization,
blas, openmp, loops placing.  
4. Science oriented guys can study SimRank.  

## Решение

1. Реализована [программа](matpow.c) выполняющая распараллеленную генерацию и возведение матриц в степень. Матрицу можно интерпретировать в качестве матрицы смежности произвольного ориентированного графа.  
    Сборка:  
    `g++ -O3 matpow.c -o run -fopenmp`  

    Запуск:  
    ./run N k n_threads need_print  
    N - размер генерируемой матрицы NxN (default = 3),  
    k - степень, в которую возводится матрица (default = 3),  
    n_threads - число тредов (default = 4),  
    need_print - флаг для вывода матрицы (default = True).  

2. Реализована [программа](pagerank.c) выполняющая распараллеленную генерацию графа и вычисление алгоритмов PageRank и NaiveRank и их сравнение.  
    Сборка:  
    `g++ -O3 pagerank.c -o run -fopenmp`  

    Запуск:  
    ./run N k n_threads  
    N - размер NxN генерируемой матрицы смежности ориентированного графа (default = 5),  
    k - максимальное число однонаправленных мультиребер графа (default = 1),  
    n_threads - число тредов (default = 4).  

3. В приведенных выше программах, используются все указанные способы повышения производительности: сборка с оптимизацией 3, использование библиотеки openmp и правильное расположение циклов.  

#pragma once

#include <mpi.h>
#include <iostream>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <time.h>
#include <cstdlib> 
#include <cmath>   

#define B1 1
#define KiB1 1024
#define MiB1 1048576
#define GiB1 1073741824


#define ALIGNMENT (sysconf(_SC_PAGESIZE))
static void* malloc_align(size_t size){
    void *p = NULL;
    int ret = posix_memalign(&p, ALIGNMENT, size);
    if (ret != 0){
        fprintf(stderr, "Failed to allocate memory on rank\n");
        exit(-1);
    }
    return p;
}

static double rand_expo(double mean)
{
    double lambda = 1.0 / mean;
    double u = rand() / (RAND_MAX + 1.0);
    return -log(1 - u) / lambda;
}

/*sleep seconds given as double*/
static int dsleep(double t)
{
    struct timespec t1, t2;
    t1.tv_sec = (long)t;
    t1.tv_nsec = (t - t1.tv_sec) * 1000000000L;
    return nanosleep(&t1, &t2);
}

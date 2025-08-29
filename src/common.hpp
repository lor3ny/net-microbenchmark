#pragma once

#include <mpi.h>
#include <iostream>
#include <unistd.h>
#include <cstring>
#include <vector>
#include <thread>
#include <chrono>

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
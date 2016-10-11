#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <errno.h>
#include <unistd.h>		// for getopt support
#include <getopt.h>		// for getopt_long support
#include <math.h>
#include <vector>
#include <sys/time.h>
#include <assert.h>

#include <lapacke/lapacke.h> // LAPACKE/LAPACK/BLAS C library

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <cusolverDn.h>


//__global__ void cagf(float *d_real, float *d_imag, float *d_arg);

//__global__ void cagf(float *d_real, float *d_imag, float *d_arg, int num_elements);

__global__ void parameterPrediction(float *d_locx, float *d_locy, float *d_param, float xhatx, float xhaty, int num_elements);

__global__ void covariancePrediction(float *d_locx, float *d_locy, float *d_param, float *d_covx, float *d_covy, float xhatx, float xhaty, int num_elements);

__global__ void covariancePrediction(float *d_locx, float *d_locy, float *d_param, float *d_cov, float xhatx, float xhaty, int num_elements);

__global__ void subtract(float *d_a, float *d_b, float *d_c, int num_elements);

__global__ void semiMajMin(float *d_eigenvalues, float k, float *d_semi, int num_ellipses);

void displayCmdUsage(void);

void checkCmdArgs(int argc, char **argv);

void parseConfig(void);

void errCheck(cudaError_t cudaError);

void checkDeviceProperties();

void generateScenario();

void cpuGeolocation();

void cudaGeolocation();

int main(int argc, char **argv);

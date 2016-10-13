#include "cuGEO.h"


/*
 *
 */
__global__ void parameterPrediction(float *d_locx, float *d_locy, float *d_param, float xhatx, float xhaty, int num_elements) {
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < num_elements) {
		d_param[tid] = atan2f((xhaty-d_locy[tid]),(xhatx-d_locx[tid]));
	}
}


/*
 * Version of the parameterPrediction kernel that uses shared memory
 * Requires amount of shared memory 2 * num_elements
 */
__global__ void parameterPredictionSmem(float *d_locx, float *d_locy, float *d_param, float xhatx, float xhaty, int num_elements) {
	extern __shared__ float smem[];
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < num_elements) {
		smem[tid] = xhatx-d_locx[tid];
		smem[tid+num_elements] = xhaty-d_locy[tid];
		__syncthreads();
		d_param[tid] = atan2f(smem[tid+num_elements],smem[tid]);
	}
}


/*
 *
 */
__global__ void covariancePrediction(float *d_locx, float *d_locy, float *d_param, float *d_cov, float xhatx, float xhaty, int num_elements) {
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < num_elements) {
		d_cov[tid] = -sin(d_param[tid]) * (1.0/pow(((xhatx-d_locx[tid])*(xhatx-d_locx[tid])+(xhaty-d_locy[tid])*(xhaty-d_locy[tid])),(float)0.5));
		d_cov[tid+num_elements] = cos(d_param[tid]) * (1.0/pow(((xhatx-d_locx[tid])*(xhatx-d_locx[tid])+(xhaty-d_locy[tid])*(xhaty-d_locy[tid])),(float)0.5));
	}
}


/*
 * Version of the covariancePrediction kernel that uses shared memory to store the output of each atomic operation
 * Requires 3 * num_elements amount of shared memory
 * TODO in old version of this, is global memory used to store each intermediate atomic op output?
 * TODO try version of this with each grouping, the ones that can happen simultaneously and those that can't, broken out as their own kernels using streams
 */
__global__ void covariancePredictionSmem(float *d_locx, float *d_locy, float *d_param, float *d_cov, float xhatx, float xhaty, int num_elements) {
	extern __shared__ float smem[];
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < num_elements) {

		smem[tid] = xhatx-d_locx[tid];						// these two lines are mutually exclusive
		smem[tid+num_elements] = xhaty-d_locy[tid];

		smem[tid] = smem[tid] * smem[tid];					// these two lines are mutually exclusive
		smem[tid+num_elements] = smem[tid+num_elements] * smem[tid+num_elements];

		smem[tid] = smem[tid] + smem[tid+num_elements];		// merges back into only using shared memory of length num_elements

		smem[tid] = pow(smem[tid], (float)0.5);				// these three lines are mutually exclusive
		smem[tid+num_elements] = -sin(d_param[tid]);
		smem[tid+2*num_elements] = cos(d_param[tid]);

		smem[tid] = 1.0 / smem[tid];

		//__syncthreads();			//TODO is this needed? even though we're writing and reading from same location, it's the same thread ID per location, so it would just execute the lines here sequentially right?
		d_cov[tid] = smem[tid+num_elements] * smem[tid];				//TODO if these two lines read from same memory, does each thread execute in order, or is this a race condition because each line gets executed by different threads?....
		d_cov[tid+num_elements] = smem[tid+2*num_elements] * smem[tid];
	}
}


/*
 *
 */
__global__ void subtract(float *d_a, float *d_b, float *d_c, int num_elements) {
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < num_elements) {
		d_c[tid] = d_a[tid] - d_b[tid];
	}
}


/*
 * Version of the subtract kernel that uses shared memory
 * Requires 2 * num_elements amount of shared memory
 */
__global__ void subtractSmem(float *d_a, float *d_b, float *d_c, int num_elements) {
	extern __shared__ float smem[];
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < num_elements) {
		smem[tid] = d_a[tid];
		smem[tid+num_elements] = d_b[tid];
		d_c[tid] = smem[tid] - smem[tid+num_elements];
	}
}


/*
 * d_eigenvalues comes as a Nx2 matrix where each row is the [max, min] eigenvalue of 1 ellipse, but
 * it's re-factored as a 2N array in column major order for CUDA, so that the first half are the max
 * values and the second half are the min values
 */
__global__ void semiMajMin(float *d_eigenvalues, float k, float *d_semi, int num_ellipses) {
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < num_ellipses) {
		d_semi[tid] = sqrt(k*d_eigenvalues[tid]);
		d_semi[tid+num_ellipses] = sqrt(k*d_eigenvalues[tid+num_ellipses]);
	}
}


/*
 * Version of the semiMajMin kernel that uses shared memory
 * Requires 2 * num_ellipses amount of shared memory
 */
__global__ void semiMajMinSmem(float *d_eigenvalues, float k, float *d_semi, int num_ellipses) {
	extern __shared__ float smem[];
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < num_ellipses) {
		smem[tid] = k * d_eigenvalues[tid];
		smem[tid+num_ellipses] = k*d_eigenvalues[tid+num_ellipses];
		d_semi[tid] = sqrt(smem[tid]);
		d_semi[tid+num_ellipses] = sqrt(smem[tid+num_ellipses]);
	}
}


/* Calculate "argument" (d_arg) or angle of a complex number (d_real, d_imag)
 * where "z = x + y*I" is equivalent to "theta = arctan(y/x)"
 * In C/C++ this would typically use glibc to perform an arctan function,
 * in CUDA there is an identical function supported
 */
/*__global__ void cagf(float *d_real, float *d_imag, float *d_arg) {
	const unsigned int tid = threadIdx.x;
	d_arg[tid] = atan2f(d_imag[tid], d_real[tid]);
}*/


/* Calculate "argument" (d_arg) or angle of a complex number (d_real, d_imag)
 * where "z = x + y*I" is equivalent to "theta = arctan(y/x)"
 * This version will use more than one CUDA grid/block
 */
/*__global__ void cagf(float *d_real, float *d_imag, float *d_arg, int num_elements) {
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < num_elements) {
		d_arg[tid] = atan2f(d_imag[tid], d_real[tid]);
	}
}*/


#include "cuGEO.h"


__global__ void parameterPrediction(float *d_locx, float *d_locy, float *d_param, float xhatx, float xhaty, int num_elements) {
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < num_elements) {
		d_param[tid] = atan2f((xhaty-d_locy[tid]),(xhatx-d_locx[tid]));
	}
}


__global__ void covariancePrediction(float *d_locx, float *d_locy, float *d_param, float *d_covx, float *d_covy, float xhatx, float xhaty, int num_elements) {
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < num_elements) {
		d_covx[tid] = -sin(d_param[tid]) * (1.0/pow(((xhatx-d_locx[tid])*(xhatx-d_locx[tid])+(xhaty-d_locy[tid])*(xhaty-d_locy[tid])),(float)0.5));
		d_covy[tid] = cos(d_param[tid]) * (1.0/pow(((xhatx-d_locx[tid])*(xhatx-d_locx[tid])+(xhaty-d_locy[tid])*(xhaty-d_locy[tid])),(float)0.5));
	}
}


__global__ void covariancePrediction(float *d_locx, float *d_locy, float *d_param, float *d_cov, float xhatx, float xhaty, int num_elements) {
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < num_elements) {
		d_cov[tid] = -sin(d_param[tid]) * (1.0/pow(((xhatx-d_locx[tid])*(xhatx-d_locx[tid])+(xhaty-d_locy[tid])*(xhaty-d_locy[tid])),(float)0.5));
		d_cov[tid+num_elements] = cos(d_param[tid]) * (1.0/pow(((xhatx-d_locx[tid])*(xhatx-d_locx[tid])+(xhaty-d_locy[tid])*(xhaty-d_locy[tid])),(float)0.5));
	}
}


__global__ void subtract(float *d_a, float *d_b, float *d_c, int num_elements) {
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < num_elements) {
		d_c[tid] = d_a[tid] - d_b[tid];
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


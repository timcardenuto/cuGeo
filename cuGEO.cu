/*
 * To compile with cuBLAS and LAPACKE/LAPACK/BLAS C libraries
 * 	nvcc cuGEO.cu kernels.cu -o cuGEO -lcublas -lcusolver -llapacke -llapack -lblas -lpython2.7 -lsatlas
 *
 *
*/

#include "cuGEO.h"

#define DEBUG 0
#define CMEMSIZE 1024	// has to be predetermined, realistically is only good for a small number of constants (duh), not regular data - this is just here for "test"

// command line args
static long int device = 0;
static long int blocks = 1;
static long int threads = 3;
static char *file;
static char *config;
static long int iterations = 1;
static long int measurements = 3;
static int verbose = 0;
static int vv = 0;
static int benchmark = 0;
static int plot = 0;
static int usesmem = 0;
static int usecmem = 0;
static int blocksflag = 0;
static int threadsflag = 0;

// cuda device properties
static int maxThreadsPerBlock = 0;

// used to print report
FILE *fpreport;
FILE *geofile;
FILE *benchmarkfile;

// scenario data
static float xhatx = 0.0;		// target location guess
static float xhaty = 0.0;
static std::vector<float> ax;	// location where measurements were taken
static std::vector<float> ay;
//static std::vector<float> z;	// measurement (for DOA = angle in radians)
static float sigma = 0.0;		// variance of measurements, assume single sensor
static int n = 3;	// n is dimensions of sigma matrix R
static float *R;				// measurement variance matrix


// benchmarking variables
float fastparam = 1.0, avgparam = 0.0;
int fastparamblock = 0, fastparamthread = 0;
float fastcov = 1.0, avgcov = 0.0;
int fastcovblock = 0, fastcovthread = 0;
float fastsemi = 1.0, avgsemi = 0.0;
int fastsemiblock = 0, fastsemithread = 0;
float fastsub = 1.0, avgsub = 0.0;
int fastsubblock = 0, fastsubthread = 0;
int fastcuexecs = 100, worstcuexecs = 0, fastcuexecblock = 0, fastcuexecthread = 0, worstcuexecblock = 0, worstcuexecthread = 0, avgcuexecs = 0;
unsigned long  fastcuexecus = 1000000000, worstcuexecus = 0, avgcuexecus = 0;
int fastcexecs = 100, worstcexecs = 0, avgcexecs = 0;
unsigned long  fastcexecus = 1000000000, worstcexecus = 0, avgcexecus = 0;

/* help message block */
void displayCmdUsage() {
	puts("Usage: ./cuGEO [OPTION] \n\
	-d	--device	Specify which GPU to use, defaults to 0. \n\
	-b	--blocks	Number of blocks to use. Does not need to be specified, will auto-distribute \n\
				based on number of measurements. \n\
	-t	--threads	Number of threads to assign per block. Does not need to be specified, will \n\
				auto-distribute based on number of measurements. \n\
	-f	--file		Path to data file that will be processed, default is auto-generated random \n\
				data. NOTE: the only currently supported file types are CSV containing DOA\n\
				measurements in the first column, the 'x' parameter of the measurement \n\
				location in the second column and the 'y' parameter of the measurement \n\
				location in the third column. The last row contains sigma values (assumes\n\
				single sensor) \n\
	-m	--measurements	Number of measurements to include in the geolocation calculation \n\
		--iterations	Number of times to re-run a single instance, used to help determine average\n\
				execution times. !!Will stack with --benchmark!! \n\
		--benchmark	Tests execution times starting at 3 data points and doubling each run until\n\
				the number of --measurements is reached \n\
	-p	--plot		Plots the geolocations, target locations and ellipses \n\
	-s	--smem		Uses shared memory kernels for comparison \n\
	-c	--cmem		Uses constant memory kernels for comparison\n\
	-x	--configure	Path to configuration file. Any parameters specified on the command line \n\
				will override the equivalent ones in this file \n\
	-v	--verbose	Prints additional output \n\
		--vv		Additional debug type information, including the output of EVERY operation \n\
		--help		Display this message \n");
	exit(1);
}

/* Use getopt to read cmd line arguments */
void checkCmdArgs(int argc, char **argv) {
    int c;
	char *ptr;
   	while (1) {
        int option_index = 0;

        static struct option long_options[] = {
			{"device",	required_argument, 	0, 'd'},
			{"blocks",	required_argument, 	0, 'b'},
			{"threads",	required_argument, 	0, 't'},
			{"file",	required_argument, 	0, 'f'},
			{"measurements",	required_argument, 	0, 'm'},
			{"iterations",	required_argument, 	0, 'i'},
			{"configure",	required_argument, 	0, 'x'},
			{"vv",	no_argument, 	&vv, 1},
			{"verbose",	no_argument, 	&verbose, 1},
			{"benchmark",	no_argument, 	&benchmark, 1},
			{"smem",	no_argument, 	&usesmem, 1},
			{"cmem",	no_argument, 	&usecmem, 1},
			{"plot",	no_argument, 	&plot, 1},
			{"help",	no_argument, 		0, 'h'},
			{0,			0, 					0, 0},
		};

		c = getopt_long_only(argc, argv, "d:b:t:f:c:m:i:x:hvsc", long_options, &option_index);

		if (c == -1) {
            break;
		}
       	switch (c) {
			 case 0:
				/* If this option set a flag, do nothing else now. */
				if (long_options[option_index].flag != 0) {
					break;
				}
				printf ("option %s", long_options[option_index].name);
				if (optarg) {
					printf (" with arg %s", optarg);
				}
				printf ("\n");
				break;
			case 'd':
				device = strtoul(optarg, &ptr, 10);
				if (strcmp(ptr,"")) {
					printf("Value %s of option %s is not a number \n", ptr, long_options[option_index].name);
					exit(1);
				}
		        break;
			case 'b':
				blocks = strtoul(optarg, &ptr, 10);
				blocksflag = 1;
				if (strcmp(ptr,"")) {
					printf("Value %s of option %s is not a number \n", ptr, long_options[option_index].name);
					exit(1);
				}
		        break;
       		case 't':
				threads = strtoul(optarg, &ptr, 10);
				threadsflag = 1;
				if (strcmp(ptr,"")) {
					printf("Value %s of option %s is not a number \n", ptr, long_options[option_index].name);
					exit(1);
				}
		        break;
       		case 'm':
				measurements = strtoul(optarg, &ptr, 10);
				if (strcmp(ptr,"")) {
					printf("Value %s of option %s is not a number \n", ptr, long_options[option_index].name);
					exit(1);
				}
		        break;
       		case 'i':
				iterations = strtoul(optarg, &ptr, 10);
				if (strcmp(ptr,"")) {
					printf("Value %s of option %s is not a number \n", ptr, long_options[option_index].name);
					exit(1);
				}
		        break;
       		case 'f':
				file = optarg;
		        break;
       		case 'x':
				config = optarg;
				parseConfig();
		        break;
       		case 'v':
				verbose = 1;
		        break;
       		case 's':
				usesmem = 1;
				break;
       		case 'c':
				usecmem = 1;
				break;
       		default:
				displayCmdUsage();
	    }
	}
	if (optind < argc) {
      	printf ("Unrecognized options: ");
      	while (optind < argc) {
        	printf ("%s ", argv[optind++]);
      	}
		printf("\n"); //putchar ('\n');
		displayCmdUsage();
    }
	return;
}


void parseConfig() {
	printf("Reading config file %s\n", config);
	FILE *fpcfg;
	if(config) {
		fpcfg = fopen(config, "r");
		if (fpcfg==NULL) {
			fprintf(stderr,"Error, failed to open file %s with error %d \n", config, errno);
			exit(1);
		}
		char line [256];
		char *pch;
		char *str;
		while (fgets(line, 256, fpcfg) != NULL) {
			pch = strtok (line,"=");	// parse based on '='
			if (strcmp(pch,"device") == 0) {
				pch = strtok (NULL, "=");
				device = strtoul(pch, &str, 10);
				printf("	Device option set to %i\n", device);

			} else if (strcmp(pch,"blocks") == 0) {
				pch = strtok (NULL, "=");
				blocks = strtoul(pch, &str, 10);
				blocksflag = 1;
				printf("	Blocks option set to %i\n", blocks);

			} else if (strcmp(pch,"threads") == 0) {
				pch = strtok (NULL, "=");
				threads = strtoul(pch, &str, 10);
				threadsflag = 1;
				printf("	Threads option set to %i\n", threads);

			} else if (strcmp(pch,"file") == 0) {
				pch = strtok (NULL, "=");
				file = pch;
				printf("	File option set to %i\n", file);

			} else if (strcmp(pch,"measurements") == 0) {
				pch = strtok (NULL, "=");
				measurements = strtoul(pch, &str, 10);
				printf("	Measurements option set to %i\n", measurements);

			} else if (strcmp(pch,"iterations") == 0) {
					pch = strtok (NULL, "=");
					iterations = strtoul(pch, &str, 10);
					printf("	Iterations option set to %i\n", iterations);

			} else if (strcmp(pch,"verbose") == 0) {
				pch = strtok (NULL, "=");
				verbose = strtoul(pch, &str, 10);
				printf("	Verbose option set to %i\n", verbose);

			} else if (strcmp(pch,"benchmark") == 0) {
				pch = strtok (NULL, "=");
				benchmark = strtoul(pch, &str, 10);
				printf("	Benchmark option set to %i\n", benchmark);

			} else {
				printf("Error, unknown option %s \n",pch);
				exit(1);
			}
		}
		fclose(fpcfg);
	} else {
		printf("Error, config file %s referenced, no file found...", config);
	}
	printf("\n");
}


/*
 * Check return from CUDA commands for errors
 */
void errCheck(cudaError_t cudaError) {
	if (cudaError != cudaSuccess) {
		printf("CUDA Error %s\n",cudaError);
		exit(1);
	}
}



/*
 * Check CUDA driver, toolkit, and device properties
 */
void checkDeviceProperties() {
	// check host CUDA versions
	int cuda_runtime;
	errCheck(cudaRuntimeGetVersion(&cuda_runtime));
	int cuda_driver;
	errCheck(cudaDriverGetVersion(&cuda_driver));
	printf("\n##### Host CUDA Properties ##### \n");
	printf("Runtime Version:	%i\n", cuda_runtime);
	printf("Driver Version:		%i\n\n", cuda_driver);
	fprintf(fpreport,"\n##### Host CUDA Properties ##### \n");
	fprintf(fpreport,"Runtime Version:	%i\n", cuda_runtime);
	fprintf(fpreport,"Driver Version:		%i\n\n", cuda_driver);
	fflush(fpreport);

	// check GPU device specs
	int cuda_devices;
	errCheck(cudaGetDeviceCount(&cuda_devices));
	if (cuda_devices > 2) {
		printf("Error, cannot handle %i devices\n",cuda_devices);
		fprintf(fpreport,"Error, cannot handle %i devices\n",cuda_devices);
		fclose(fpreport);
		cudaDeviceReset();
		exit(1);
	}
	for (int i = 0; i < cuda_devices; i++) {
		cudaDeviceProp prop;
		errCheck(cudaGetDeviceProperties(&prop, i));
		maxThreadsPerBlock = prop.maxThreadsPerBlock;

		printf("\n##### GPU Device %i Properties ##### \n", i);
		printf("Device Name:		%s\n", prop.name);
		printf("Compute Capability:	%i.%i\n", prop.major, prop.minor);
		printf("Multi-Processors:	%i\n", prop.multiProcessorCount);
		printf("Threads per Warp:	%i\n", prop.warpSize);
		printf("Threads per Block:	%i\n", prop.maxThreadsPerBlock);
		printf("Registers per Block:	%i\n", prop.regsPerBlock);
		printf("Shared Mem per Block:	%i\n", prop.sharedMemPerBlock);
		printf("Constant Memory:	%i\n", prop.totalConstMem);
		printf("Global Memory:		%i\n", prop.totalGlobalMem);
		printf("Max Block Dimensions:	[%i, %i, %i]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max Grid Dimensions:	[%i, %i, %i]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("Concurrent Kernel Execution:		%i\n", prop.concurrentKernels);
		printf("Concurrent Memcpy and Kernel Execution:	%i\n", prop.deviceOverlap);
		printf("\n");
		// print to file
		fprintf(fpreport,"\n##### GPU Device %i Properties ##### \n", i);
		fprintf(fpreport,"Device Name:		%s\n", prop.name);
		fprintf(fpreport,"Compute Capability:	%i.%i\n", prop.major, prop.minor);
		fprintf(fpreport,"Multi-Processors:	%i\n", prop.multiProcessorCount);
		fprintf(fpreport,"Threads per Warp:	%i\n", prop.warpSize);
		fprintf(fpreport,"Threads per Block:	%i\n", prop.maxThreadsPerBlock);
		fprintf(fpreport,"Registers per Block:	%i\n", prop.regsPerBlock);
		fprintf(fpreport,"Shared Mem per Block:	%i\n", prop.sharedMemPerBlock);
		fprintf(fpreport,"Constant Memory:	%i\n", prop.totalConstMem);
		fprintf(fpreport,"Global Memory:		%i\n", prop.totalGlobalMem);
		fprintf(fpreport,"Max Block Dimensions:	[%i, %i, %i]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		fprintf(fpreport,"Max Grid Dimensions:	[%i, %i, %i]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		fprintf(fpreport,"Concurrent Kernel Execution:		%i\n", prop.concurrentKernels);
		fprintf(fpreport,"Concurrent Memcpy and Kernel Execution:	%i\n", prop.deviceOverlap);
		fprintf(fpreport,"\n");
		fflush(fpreport);
	}
}



/*
 * Generate truth data on target locations, measurement locations, measurement angles over time
 * Add error to measurements to use in Geo calculation
 */
float* generateScenario() {
	if (verbose) { 	printf("\n##### Generating Geo Scenario #####\n");
					fprintf(fpreport,"\n##### Generating Geo Scenario #####\n"); }

	// TODO: write MATLAB script to autogenerate locations/measurements/errors for a scenario

	ax.clear();
	ay.clear();
	//z.clear();

	//TODO need to turn total number of measurements into sensical block/thread sets, aka 100,000 threads and 1 block will fail.
	n = measurements;	// this will allow the rest of the code to actually try with the user argument. If using a file, only use "measurement" number of rows

	float *z = (float *)malloc(sizeof(float) * n);

	// builds R sensor variance identity matrix based on number of measurements
	// TODO is this correct? either it's a scalar b/c all measurements have same variance, or it's an identity of size NxN b/c otherwise the math wouldn't work out right
	R = (float *)malloc(sizeof(float)*n*n);
	sigma = 2.0/180.0*M_PI;		// variance of measurements, TODO assumes single sensor or all the same variance
	int count = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			//printf("R[%i]	",i*n+j);
			if (j==count) {
				R[i*n+j] = sigma*sigma;
			} else {
				R[i*n+j] = 0.0;
			}
		}
		count++;
		//printf("\n");
	}

	//TODO pull R from file
	if (file) {	// if we're using a data file with DOA measurements
		FILE *fp;
		fp = fopen(file, "r");
		if (fp==NULL) {
			fprintf(stderr,"Failed to open file %s with error %d \n", file, errno);
			fprintf(fpreport,"Failed to open file %s with error %d \n", file, errno);
			fclose(fpreport);
			cudaDeviceReset();
			exit(1);
		}
		int filesize;
		fseek(fp, 0L, SEEK_END);
		filesize = ftell(fp);
		rewind(fp);
		printf("File Size:		%i \n", filesize);
		fprintf(fpreport,"File Size:		%i \n", filesize);

		printf("Loading DOA measurements from file...\n");
		fprintf(fpreport,"Loading DOA measurements from file...\n");

		char line[1024];
		while (fgets(line, 1024, fp))
		{
			char* tmp = strdup(line);
			const char* tok;
			int num = 1;
			for (tok = strtok(tmp, ",");
					tok && *tok;
					tok = strtok(NULL, ",\n"))
			{
				/*
				if (!--num) {
					z.push_back(strtof(tok,NULL));
				} else if (!--num) {
					ax.push_back(strtof(tok,NULL));
				} else if (!--num) {
					ay.push_back(strtof(tok,NULL));
				}
				n++;
				*/
			}
			// NOTE strtok clobbers tmp
			free(tmp);
		}
		fclose(fp);


	} else {	// generate some random DOA measurements
		ax.push_back((sin(60.0/180.0*M_PI)*15.0*1852.0));	// measurement location 1
		ay.push_back((cos(60.0/180.0*M_PI)*15.0*1852.0));
		//z.push_back(-(90.0+60.0)/180.0*M_PI);				// measurement DOA 1
		z[0] = -(90.0+60.0)/180.0*M_PI;

		for (int i = 0; i < n-2; i++) {
			ax.push_back((sin(-5.0/180.0*M_PI)*10.0*1852.0));	// measurement location 2
			ay.push_back((cos(5.0/180.0*M_PI)*10.0*1852.0));
			//z.push_back(-(90.0-5.0)/180.0*M_PI);				// measurement DOA 2
			z[1+i] = -(90.0-5.0)/180.0*M_PI;
		}

		ax.push_back((sin(10.0/180.0*M_PI)*5.0*1852.0));	// measurement location 3
		ay.push_back((cos(10.0/180.0*M_PI)*5.0*1852.0));
		//z.push_back(-(90.0+10.0)/180.0*M_PI);				// measurement DOA 3
		z[n-1] = -(90.0+10.0)/180.0*M_PI;
	}

	// target location guess
	xhatx = 2.0*1852.0;
	xhaty = 2.0*1852.0;

	if (verbose) { printf("Data set size: %i\n", measurements);
					printf("N dimension size: %i\n", n); }

	if (DEBUG || vv) {	// print z, x, a
		printf("measured data z =\n");
		for (int i = 0; i < n; i++) {
			printf("	%f\n",z[i]);
		}
		printf("\n");
		printf("measurement error R =\n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				printf("	%f",R[i*n+j]);
			}
			printf("\n");
		}
		printf("\n");
		printf("measurement location a =\n");
		for (int i = 0; i < n; i++) {
			printf("	%f	%f\n",ax[i],ay[i]);
		}
		printf("\n");
	}
	if (verbose) { 	printf("target location guess xhat =\n");
					printf("	%f	%f\n",xhatx,xhaty);
					printf("\n"); }
	return z;
}



/*
 * C/C++ version of DOA Geolocation algorithm using BLAS, LAPACK
 */
void cpuGeolocation(float *z) {

	if (true) { 	printf("\n##### Executing C/C++ Geo Routine #####\n");
					fprintf(fpreport, "##### Executing C/C++ Geo Routine #####\n"); }

	struct timespec tp1, tp2;
	clock_gettime(CLOCK_REALTIME, &tp1);

	// predict next parameter state (h)
	if (verbose) { 	printf("Predicting next parameter state (h)...\n");
					fprintf(fpreport,"Predicting next parameter state (h)...\n"); }
	std::vector<float> h;
	for (int i = 0; i < n; i++) {
		h.push_back(atan2f((xhaty-ay[i]),(xhatx-ax[i])));
	}
	
	// predict next state Covariance (H)
	if (verbose) { 	printf("Predicting next state Covariance (H)...\n");
					fprintf(fpreport,"Predicting next state Covariance (H)...\n"); }
	float* H = (float *)malloc(sizeof(float)*n*2);
	for (int i = 0; i < n; i++) {
		H[i] = (-sin(h[i]) * (1.0/pow(((xhatx-ax[i])*(xhatx-ax[i])+(xhaty-ay[i])*(xhaty-ay[i])),0.5))); //x
		H[i+n] = (cos(h[i]) * (1.0/pow(((xhatx-ax[i])*(xhatx-ax[i])+(xhaty-ay[i])*(xhaty-ay[i])),0.5))); //y
	}

	if (DEBUG || vv) {	// print h, H
		printf("	h =\n");
		for (int i = 0; i < n; i++) {
			printf("	%f\n",h[i]);
		}
		printf("\n");
		printf("	H =\n");
		for (int i = 0; i < n; i++) {
			printf("	%f	%f\n", H[i], H[i+n]);
		}
		printf("\n");
	}

	// calculate inverse of R
	if (verbose) { 	printf("Calculating inverse of R: inv(%ix%i)\n", n, n);
					fprintf(fpreport,"Calculating inverse of R ...\n"); }
	int INFO = 0;
	int N = n;
	int M = n;
	int LDA = n; // leading dimension of array A
	int *IPIV = (int *)malloc(sizeof(float)*N);
	float *tempInvR = (float *)malloc(sizeof(float)*LDA*N);
	/*
	int LWORK = n;
	for (int i = 0; i < n*n; i++) {
		tempInvR[i] = R[i];
	}
	sgetrf_(&M, &N, tempInvR, &LDA, IPIV, &INFO);
	if (INFO != 0) {
		printf("Error during matrix inversion\n");
		fprintf(fpreport,"Error during matrix inversion\n");
		exit(1);
	}
	float *WORK = (float *)malloc(sizeof(float)*LWORK);
	sgetri_(&N, tempInvR, &LDA, IPIV, WORK, &LWORK, &INFO);
	if (INFO != 0) {
		printf("Error during matrix inversion\n");
		fprintf(fpreport,"Error during matrix inversion\n");
		exit(1);
	}
	free(IPIV);
	free(WORK);
	*/

	int count = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			//printf("R[%i]	",i*n+j);
			if (j==count) {
				tempInvR[i*n+j] = 1.0/R[0];
			} else {
				tempInvR[i*n+j] = 0.0;
			}
		}
		count++;
		//printf("\n");
	}

	if (DEBUG || vv) {	// check inv(R) calculation
		printf("	inv(R) = \n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				printf("	%f",tempInvR[i*n+j]);
			}
			printf("\n");
		}
	}

	// H'*inv(R)
	float *tempA = (float *)malloc(sizeof(float)*2*n);
	cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,2,n,n,1.0,H,n,tempInvR,n,0.0,tempA,2);

	if (DEBUG || vv) {
		printf("	H'*inv(R) = \n");
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < n; j++) {
				printf("	%f",tempA[i+j*2]);
			}
			printf("\n");
		}
	}

	// H'*inv(R)*H
	float *tempB = (float *)malloc(sizeof(float)*2*2);
	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,2,2,n,1.0,tempA,2,H,n,0.0,tempB,2);

	if (DEBUG || vv) {
		printf("	H'*inv(R)*H = \n");
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				printf("	%f",tempB[i+j*2]);
			}
			printf("\n");
		}
	}

	free(tempA);


	// P = inv(H'*inv(R)*H)
	N = 2;
	M = 2;
	LDA = 2;
	int *ipiv = (int *)malloc(sizeof(float)*N);
	int lwork = n;
	sgetrf_(&M, &N, tempB, &LDA, ipiv, &INFO);
	if (INFO != 0) {
		printf("Error during matrix inversion\n");
		fprintf(fpreport,"Error during matrix inversion\n");
		exit(1);
	}
	float *work = (float *)malloc(sizeof(float)*lwork);
	sgetri_(&M, tempB, &LDA, ipiv, work, &lwork, &INFO);
	if (INFO != 0) {
		printf("Error during matrix inversion\n");
		fprintf(fpreport,"Error during matrix inversion\n");
		exit(1);
	}

	if (DEBUG || vv) {	// check inv(R) calculation
		printf("	P = \n");
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				printf("	%f",tempB[i+j*2]);
			}
			printf("\n");
		}
	}
	free(ipiv);
	free(work);

	// P*H'
	float *tempC = (float *)malloc(sizeof(float)*2*n);
	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,2,n,2,1.0,tempB,2,H,n,0.0,tempC,2);

	if (DEBUG || vv) {
		printf("	P*H' = \n");
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < n; j++) {
				printf("	%f",tempC[i+j*2]);
			}
			printf("\n");
		}
	}

	// P*H'*inv(R)
	float *tempD = (float *)malloc(sizeof(float)*2*n);
	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,2,n,n,1.0,tempC,2,tempInvR,n,0.0,tempD,2);

	if (DEBUG || vv) {
		printf("	P*H'*inv(R) = \n");
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < n; j++) {
				printf("	%f",tempD[i+j*2]);
			}
			printf("\n");
		}
	}
	free(tempC);

	// z - h
	float* tempE = (float *)malloc(sizeof(float)*n);
	for (int i = 0; i < n; i++) {
		tempE[i] = z[i] - h[i];
	}

	// P*H'*inv(R)*(z-h)
	float *tempF = (float *)malloc(sizeof(float)*2);
	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,2,1,n,1.0,tempD,2,tempE,n,0.0,tempF,2);

	if (DEBUG || vv) {
		printf("	P*H'*inv(R)*(z-h) = \n");
		printf("	%f	%f\n",tempF[0],tempF[1]);
	}
	free(tempD);
	free(tempE);

	// xhat + P*H'*inv(R)*(z-h)
	xhatx = xhatx + tempF[0];
	xhaty = xhaty + tempF[1];
	if (verbose) { 	printf("	xhat = %f, %f\n", xhatx, xhaty);
					fprintf(fpreport,"	xhat = %f, %f\n", xhatx, xhaty); }
	free(tempF);


	// calculate eig(P)
	char jobu = 'A'; // don't need eigenvectors so don't calculate columns of U or VT
	char jobvt = 'A';
	float* U = (float *)malloc(sizeof(float)*2*2);
	float* VT = (float *)malloc(sizeof(float)*2*2);
	M = 2;
	N = 2;
	LDA = 2;
	int LDU = 2;
	int LDVT = 2;
	float Eig[2];
	lwork = 272;
	float* ework = (float *)malloc(sizeof(float)*lwork);
	sgesvd_(&jobu, &jobvt, &M, &N, tempB, &LDA, Eig, U, &LDU, VT, &LDVT, ework, &lwork, &INFO);

	if (DEBUG || vv) {
		printf("	eig(P) = \n");
		printf("	%f, %f\n",Eig[0],Eig[1]);
	}


	free(tempInvR);
	free(tempB);

	// calculate semimajor and semiminor values
	// TODO handle more than 1 ellipse
	if (verbose) { 	printf("\nCalculating 95%% semimajor, semiminor ellipse bounds \n");
					fprintf(fpreport,"\nCalculating 95%% semimajor, semiminor ellipse bounds \n");	}
	int num_ellipses = 1;
	float Emem_size = sizeof(float)*2*num_ellipses;
	float k = 5.9915;	// magic constant for 95% confidence ellipse
	float *semi = (float *)malloc(Emem_size);
	semi[0] = sqrt(k*Eig[0]);
	semi[1] = sqrt(k*Eig[1]);
	if (verbose) { 	printf("	semimajor = %f\n",semi[0]);			// semimajor is always first, output of calculation orders values based on magnitude
					printf("	semiminor = %f\n",semi[1]);
					printf("\n");
					fprintf(fpreport,"	semimajor = %f\n",semi[0]);
					fprintf(fpreport,"	semiminor = %f\n",semi[1]);
					fprintf(fpreport,"\n"); }
	free(semi);


	clock_gettime(CLOCK_REALTIME, &tp2);
	time_t s2;
	unsigned long us2;
	if (tp2.tv_nsec < tp1.tv_nsec) {
		s2 = (tp2.tv_sec-tp1.tv_sec) - 1;
		us2 = (tp2.tv_nsec + (1000000000 - tp1.tv_nsec)) / 1000;
	} else {
		s2 = tp2.tv_sec-tp1.tv_sec;
		us2 = (tp2.tv_nsec-tp1.tv_nsec) / 1000;
	}

	avgcexecs = avgcexecs + s2;
	avgcexecus = avgcexecus + us2;
	if (s2 < fastcexecs || (s2 == fastcexecs && us2 < fastcexecus)) {
		fastcexecs = s2;
		fastcexecus = us2;
	} else if (s2 > worstcexecs || (s2 == worstcexecs && us2 > fastcexecus)) {
		worstcexecs = s2;
		worstcexecus = us2;
	}

	if (true) { 	printf("C/C++ Geolocation execution time: %i s  %lu us\n", (int)s2, us2);
					fprintf(fpreport,"C/C++ Geolocation execution time: %i s  %lu us\n", (int)s2, us2);
					printf("\n");
					fprintf(fpreport,"\n");
					fflush(fpreport); }

	fprintf(benchmarkfile, "%i,%lu,", (int)s2, us2);
}



/*
 * Version of the parameterPrediction kernel that uses constant memory - it won't work unless in the same file as the CPU call...
 * Puts input data into constant memory (cmem) instead of global memory - probably will not be more efficient since constant memory is
 * intended for a small amount of data needed by *every* thread
 * It *does* make sense to use constant memory for the xhatx, xhaty, and num_elements values because they are accessed by every thread
 */
__constant__ float cmem[CMEMSIZE];
__global__ void parameterPredictionCmem(float *d_param, float xhatx, float xhaty, int num_elements) {
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < num_elements) {
		d_param[tid] = atan2f((xhaty-cmem[tid+num_elements]),(xhatx-cmem[tid]));
	}
}



/*
 * CUDA version of DOA Geolocation algorithm using cuBLAS, cuSPARSE
 */
void cudaGeolocation(float *z) {
	//###############################################################################################
	// CUDA version of DOA Geolocation ---------------------------------------------------
	if (verbose) { 	printf("\n##### Executing CUDA Geo Routine #####\n");
					fprintf(fpreport,"\n##### Executing CUDA Geo Routine #####\n"); }

	struct timespec tp1, tp2;
	clock_gettime(CLOCK_REALTIME, &tp1);

	// TODO Move all memcpy that can be to beginning here, and attempt to re-use if possible (double buffering scheme), don't "free" every temp buffer just overwrite them
	// TODO Add streams where I can do more than one thing at a time...

	cublasHandle_t cublas_handle;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cublas_status = cublasCreate(&cublas_handle);
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);
	cusolverDnHandle_t cusolver_handle = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cusolver_status = cusolverDnCreate(&cusolver_handle);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	// allocate host memory
	if (verbose) { 	printf("Allocating host memory...\n");
					fprintf(fpreport,"Allocating host memory...\n"); }

	int num_ellipses = 1; // TODO figure out how to handle more than 1 ellipse

	unsigned int mem_size = sizeof(float) * n;			// data size
	unsigned int Rmem_size = sizeof(float) * n*n;		// might be different b/c dependent on how many unique sigma values... might have identical ones for different measurement sources
	unsigned int Hmem_size = sizeof(float) * 2*n;	// most intermediate arrays will be 2xN
	unsigned int Pmem_size = sizeof(float) * 2*2;		// P is always a 2x2 matrix
	float Emem_size = sizeof(float)*2*num_ellipses;		// TODO why float?

	float *h_locx = (float *)malloc(mem_size);
	float *h_locy = (float *)malloc(mem_size);
	float *h_Rinv = (float *)malloc(sizeof(float)*n*n);
	float *alpha = (float *)malloc(sizeof(float));
	float *beta = (float *)malloc(sizeof(float));
	float *h_temph = (float *)malloc(2*sizeof(float));
	float k = 5.9915;									// magic constant for 95% confidence ellipse
	float *h_semi = (float *)malloc(Emem_size);

	// create host data
	for (int i = 0; i < n; i++) {
		h_locx[i] =  ax[i];
		h_locy[i] =  ay[i];
	}
	if(usecmem) {
		float *h_loc = (float *)malloc(2*n*sizeof(float));
		for (int i = 0; i < n; i++) {
			h_loc[i] = h_locx[i];
			h_loc[i+n] = h_locy[i];
		}
		cudaMemcpyToSymbol(cmem, h_loc, 2*n*sizeof(float));
		free(h_loc);
	}
	alpha[0] = 1.0;
	beta[0] = 0.0;

	// allocate target GPU device memory
	if (verbose) { 	printf("Allocating device memory...\n");
					fprintf(fpreport,"Allocating device memory...\n"); }

	float *d_locx;
	float *d_locy;
	float *d_h;
	float *d_H;
	float *d_Rinv;
	float *d_tempa;
	float *d_tempb;
	int batchSize = 1;
	int h_INFO = 0;
	int *P, *INFO;
	float *d_P;
	float *d_tempf;
	float *d_temph;
	float U[4];				// eig(P) values
	float VT[4];
	float h_Eig[2];
	float *d_Eig = NULL;
	float *d_VT = NULL;
	int *devInfo = NULL;
	float *d_work = NULL;
	float *d_rwork = NULL;
	int lwork = 272;
	int info_gpu = 0;
	signed char jobu = 'A'; // don't need eigenvectors so don't calculate columns of U or VT
	signed char jobvt = 'A';
	float *d_tempi;

	cudaMalloc((void**) &d_locx, mem_size);
	cudaMalloc((void**) &d_locy, mem_size);
	cudaMalloc((void**) &d_h, mem_size);
	cudaMalloc((void**) &d_H, Hmem_size);
	cudaMalloc<float>(&d_Rinv,Rmem_size);
	cudaMalloc<float>(&d_tempa, Hmem_size);
	cudaMalloc<float>(&d_tempb, Pmem_size);
	cudaMalloc<int>(&P,n*batchSize*sizeof(int));
	cudaMalloc<int>(&INFO,batchSize*sizeof(int));
	cudaMalloc<float>(&d_P, Pmem_size);
	cudaMalloc<float>(&d_tempf, Hmem_size);
	cudaMalloc<float>(&d_temph, sizeof(float)*2);
	cudaMalloc((void**) &d_Eig, sizeof(float)*2);
	cudaMalloc((void**) &d_VT , Pmem_size);
	cudaMalloc((void**) &devInfo, sizeof(int));
	cudaMalloc((void**) &d_work , sizeof(float)*lwork);
	cudaMalloc((void**) &d_tempi, Emem_size);


	// copy host memory to device
	if (verbose) { 	printf("Copying data from host memory to GPU memory...\n");
					fprintf(fpreport,"Copying data from host memory to GPU memory\n"); }
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaMemcpy(d_locx, h_locx, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_locy, h_locy, mem_size, cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime = 0;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	if (verbose) { 	printf("Memcpy from host to GPU execution time: %f ms\n", elapsedTime);
					fprintf(fpreport,"Memcpy from host to GPU execution time: %f ms\n", elapsedTime); }

	clock_gettime(CLOCK_REALTIME, &tp2);
	time_t s1;
	unsigned long us1;
	if (tp2.tv_nsec < tp1.tv_nsec) {
		s1 = (tp2.tv_sec-tp1.tv_sec) - 1;
		us1 = (tp2.tv_nsec + (1000000000 - tp1.tv_nsec)) / 1000;
	} else {
		s1 = tp2.tv_sec-tp1.tv_sec;
		us1 = (tp2.tv_nsec-tp1.tv_nsec) / 1000;
	}

	if (true) { 	printf("CUDA Geolocation memory allocation time: %i s  %lu us\n", (int)s1, us1);
					fprintf(fpreport,"CUDA Geolocation memory allocation time: %i s  %lu us\n", (int)s1, us1);
					printf("\n");
					fprintf(fpreport,"\n");
					fflush(fpreport); }

	// Pin time for execution length
	clock_gettime(CLOCK_REALTIME, &tp1);

	//#####################################################################
	// execute parameterPrediction kernel
	if (verbose) { 	printf("\nLaunching parameterPrediction CUDA kernel\n");
					fprintf(fpreport,"\nLaunching parameterPrediction CUDA kernel\n"); }
	cudaEventRecord(start, 0);
	if (usesmem) {
		parameterPredictionSmem<<< blocks, threads, 2*n*sizeof(float) >>>(d_locx, d_locy, d_h, xhatx, xhaty, n);
	} else if (usecmem) {
		parameterPredictionCmem<<< blocks, threads >>>(d_h, xhatx, xhaty, n);
		usecmem=0; // TODO last use, set to zero so we can re-run and catch the last else kernel types
	} else {
		parameterPrediction<<< blocks, threads >>>(d_locx, d_locy, d_h, xhatx, xhaty, n);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	if (verbose) { 	printf("	parameterPrediction execution time: %f ms\n", elapsedTime);
					fprintf(fpreport,"	parameterPrediction execution time: %f ms\n", elapsedTime); }
	avgparam = avgparam + elapsedTime;
	if (elapsedTime < fastparam) {
		fastparam = elapsedTime;
		fastparamblock = blocks;
		fastparamthread = threads;
	}

	if (DEBUG || vv) {	// check CUDA h calculation
		float *h_param = (float *)malloc(mem_size);
		printf("	Copying processed data from GPU to host memory\n");
		cudaEventRecord(start, 0);
		cudaMemcpy(h_param, d_h, mem_size, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		printf("	Memcpy from GPU to host execution time: %f ms\n", elapsedTime);
		printf("	h =\n");
		for (int i = 0; i < n; i++) {
			printf("	%f\n",h_param[i]);
		}
		free(h_param);
	}

	//#####################################################################
	/* execute covariancePrediction kernel
	 * INPUT	h is Nx1
	 * OUTPUT	H is Nx2
	 */
	if (verbose) { 	printf("\nLaunching covariancePrediction CUDA kernel\n");
					fprintf(fpreport,"Launching covariancePrediction CUDA kernel\n"); }
	cudaEventRecord(start, 0);
	if(usesmem) {
		covariancePredictionSmem<<< blocks, threads, 3*n*sizeof(float) >>>(d_locx, d_locy, d_h, d_H, xhatx, xhaty, n);
	} else {
		covariancePrediction<<< blocks, threads >>>(d_locx, d_locy, d_h, d_H, xhatx, xhaty, n);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	if (verbose) { 	printf("	covariancePrediction execution time: %f ms\n", elapsedTime);
					fprintf(fpreport,"	covariancePrediction execution time: %f ms\n\n", elapsedTime); }
	avgcov = avgcov + elapsedTime;
	if (elapsedTime < fastcov) {
		fastcov = elapsedTime;
		fastcovblock = blocks;
		fastcovthread = threads;
	}

	if (DEBUG || vv) {	// check CUDA H calculation
		float *h_H = (float *)malloc(mem_size*2);
		printf("	Copying processed data from GPU to host memory\n");
		cudaEventRecord(start, 0);
		cudaMemcpy(h_H, d_H, Hmem_size, cudaMemcpyDeviceToHost);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsedTime, start, stop);
		printf("	Memcpy from GPU to host execution time: %f ms\n", elapsedTime);
		printf("	H =\n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < 2; j++) {
				printf("	%f",h_H[i*2+j]);
			}
			printf("\n");
		}
		free(h_H);
	}

	//#####################################################################
	/* execute R inversion kernel - only need to do this once
	 * TODO these functions are not intended for large matrices and will lag, matrix inversion is inefficient,
	 * since I have to build R to begin with, I could just invert each value while I build the matrix
	 * INPUT	R is NxN
	 * OUTPUT	inv(R) is NxN
	 */
	/* if (verbose) { 	printf("\nLaunching matrix inversion using cuBLAS: inv(%ix%i) \n", n, n);
					fprintf(fpreport,"\nLaunching matrix inversion using cuBLAS: inv(%ix%i) \n", n, n); }
	struct timeval tv3, tv4;
	gettimeofday(&tv3,NULL);

	int batchSize = 1; 			// always 1, there's only 1 R matrix	
	int lda = n;
	float* d_R, *d_Rinv;
	cudaMalloc<float>(&d_R,Rmem_size);
	cudaMemcpy(d_R, R, Rmem_size, cudaMemcpyHostToDevice);
	cudaMalloc<float>(&d_Rinv,Rmem_size);
	int *P, *INFO;	
	cudaMalloc<int>(&P,n*batchSize*sizeof(int));
	cudaMalloc<int>(&INFO,batchSize*sizeof(int));
	float *h_A[] = { d_R };
	float** d_A;
	cudaMalloc<float*>(&d_A,sizeof(h_A));
	cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);
	cublasSgetrfBatched(cublas_handle, n, d_A, lda, P, INFO, batchSize);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);

	if (verbose) { 	printf("	cublasSgetrfBatched execution time: %f ms\n", elapsedTime);
					fprintf(fpreport,"	cublasSgetrfBatched execution time: %f ms\n\n", elapsedTime); }

	int h_INFO = 0;
	cudaMemcpy(&h_INFO, INFO, sizeof(int), cudaMemcpyDeviceToHost);
	if(h_INFO == n) {
        fprintf(stderr, "Factorization Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
    float* C[] = { d_Rinv };
    float** d_C;
    cudaMalloc<float*>(&d_C,sizeof(C));
    cudaMemcpy(d_C, C, sizeof(C), cudaMemcpyHostToDevice);

    cudaEventRecord(start, 0);
    cublasSgetriBatched(cublas_handle, n, (const float **)d_A, lda, P, d_C, lda, INFO, batchSize);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	if (verbose) { 	printf("	cublasSgetriBatched execution time: %f ms\n", elapsedTime);
					fprintf(fpreport,"	cublasSgetriBatched execution time: %f ms\n\n", elapsedTime); }

    cudaMemcpy(&h_INFO, INFO, sizeof(int), cudaMemcpyDeviceToHost);
    if(h_INFO != 0)
    {
        fprintf(stderr, "Inversion Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    //cudaFree(P);
	//cudaFree(INFO);
	cudaFree(d_R);
	cudaFree(d_A);
	cudaFree(d_C);

	gettimeofday(&tv4,NULL);
	if (verbose) { 	printf("	Total matrix inversion time: %ds %dus\n", (tv4.tv_sec-tv3.tv_sec), (tv4.tv_usec-tv3.tv_usec));
					fprintf(fpreport,"	Total matrix inversion time: %ds %dus\n", (tv4.tv_sec-tv3.tv_sec), (tv4.tv_usec-tv3.tv_usec)); }

	if (DEBUG || vv) {	// check CUDA intermediate calculation
		float* h_Rinv = (float *)malloc(Rmem_size);
		cudaMemcpy(h_Rinv, d_Rinv, Rmem_size, cudaMemcpyDeviceToHost);
		printf("	inv(R) = \n");
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				printf("	%f",h_Rinv[i*n+j]);
			}
			printf("\n");
		}
		free(h_Rinv);
	}
	*/

	if (verbose) { 	printf("\nLaunching short matrix inversion for single sigma: inv(%ix%i) \n", n, n);
					fprintf(fpreport,"\nLaunching short matrix inversion for single sigma: inv(%ix%i) \n", n, n); }

	int count = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			//printf("R[%i]	",i*n+j);
			if (j==count) {
				h_Rinv[i*n+j] = 1.0/R[0];
			} else {
				h_Rinv[i*n+j] = 0.0;
			}
		}
		count++;
		//printf("\n");
	}

	cudaMemcpy(d_Rinv, h_Rinv, Rmem_size, cudaMemcpyHostToDevice);


	//#####################################################################
	/* H'*inv(R)
	 * INPUT	H is Nx2, R is NxN
	 * OUTPUT	H'*inv(R) is 2xN
	 * Note, args for cublasSgemm are nuanced whether they are after the operation or before the operation, LDA is leading dimension of A *before* Op, while M/N/K are *after* Op
	 * TODO If every measurement has the same sigma then it would be faster to simply multiply by the scalar
	 * TODO following DOESN'T work because it doesn't do a Transpose on d_H
	 * cublasSetVector(2*n, sizeof(float), h_H, 1, d_tempa, 1);
	 * float Rinv = (1.0/R[0]);
	 * cublas_status = cublasSscal(cublas_handle, 2*n, &Rinv, d_tempa, 1);
	 */
	if (verbose) { 	printf("\nLaunching matrix multiplication using cuBLAS: %ix2 * %ix%i \n", n, n, n);
					fprintf(fpreport,"\nLaunching matrix multiplication using cuBLAS: %ix2 * %ix%i \n", n, n, n); }

	cublas_status = cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 2, n, n, alpha, d_H, n, d_Rinv, n, beta, d_tempa, 2);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);

	if (DEBUG || vv) {	// check CUDA intermediate calculation
		float *h_tempa = (float *)malloc(Hmem_size);
		cudaMemcpy(h_tempa, d_tempa, Hmem_size, cudaMemcpyDeviceToHost);
		printf("	H'*inv(R) = \n");
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < n; j++) {
				printf("	%f",h_tempa[i+j*2]);
			}
			printf("\n");
		}
		free(h_tempa);
	}

	//#####################################################################
	/* H'*inv(R)*H
	 * INPUT	H'*inv(R) is 2xN	H is Nx2
	 * // OUTPUT	H'*inv(R)*H is 2x2
	 */
	if (verbose) { 	printf("\nLaunching matrix multiplication using cuBLAS: 2x%i * %ix2 \n", n, n);
					fprintf(fpreport,"\nLaunching matrix multiplication using cuBLAS: 2x%i * %ix2 \n", n, n); }

	cublas_status = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, n, alpha, d_tempa, 2, d_H, n, beta, d_tempb, 2);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);

	if (DEBUG || vv) {	// check CUDA intermediate calculation
		float *h_tempb = (float *)malloc(Pmem_size);
		cudaMemcpy(h_tempb, d_tempb, Pmem_size, cudaMemcpyDeviceToHost);
		printf("	H'*inv(R)*H = \n");
		printf("	%f	%f\n", h_tempb[0], h_tempb[2]);
		printf("	%f	%f\n", h_tempb[1], h_tempb[3]);
		free(h_tempb);
	}

	//#####################################################################
	/* P = inv(H'*inv(R)*H)
	 * INPUT	H'*inv(R)*H is 2x2
	 * OUTPUT	P is 2x2
	 */
	if (verbose) { 	printf("\nLaunching matrix inversion using cuBLAS: inv(2x2) \n");
					fprintf(fpreport,"\nLaunching matrix inversion using cuBLAS: inv(2x2) \n");	}

	float *h_tempc[] = { d_tempb };
	float** d_tempc;
	cudaMalloc<float*>(&d_tempc,sizeof(h_tempc));
	cudaMemcpy(d_tempc, h_tempc, sizeof(h_tempc), cudaMemcpyHostToDevice);
	cublasSgetrfBatched(cublas_handle, 2, d_tempc, 2, P, INFO, batchSize);
	cudaMemcpy(&h_INFO, INFO, sizeof(int), cudaMemcpyDeviceToHost);
	if(h_INFO == n) {
        fprintf(stderr, "Factorization Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

	float* h_tempd[] = { d_P };
	float** d_tempd;
	cudaMalloc<float*>(&d_tempd,sizeof(h_tempd));
	cudaMemcpy(d_tempd, h_tempd, sizeof(h_tempd), cudaMemcpyHostToDevice);
    cublasSgetriBatched(cublas_handle, 2, (const float **)d_tempc, 2, P, d_tempd, 2, INFO, batchSize);
    cudaMemcpy(&h_INFO, INFO, sizeof(int), cudaMemcpyDeviceToHost);
    if(h_INFO != 0) {
        fprintf(stderr, "Inversion Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

	if (DEBUG || vv) {
		float *h_P = (float *)malloc(Pmem_size);
		cudaMemcpy(h_P, d_P, Pmem_size, cudaMemcpyDeviceToHost);
		printf("	P = \n");
		printf("	%f	%f\n", h_P[0], h_P[2]);
		printf("	%f	%f\n", h_P[1], h_P[3]);
		free(h_P);
	}

	//#####################################################################
	/* P*H'
	 * INPUT	P is 2x2	H is Nx2
	 * OUTPUT	P*H' is 2xN
	 */
	if (verbose) { 	printf("\nLaunching matrix multiplication using cuBLAS: 2x2 * %ix2 \n", n, n);
					fprintf(fpreport,"\nLaunching matrix multiplication using cuBLAS: 2x2 * %ix2 \n", n, n); }

	cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 2, n, 2, alpha, d_P, 2, d_H, n, beta, d_tempa, 2);
	
	if (DEBUG || vv) {	// check CUDA intermediate calculation
		float *h_tempe = (float *)malloc(Hmem_size);
		cudaMemcpy(h_tempe, d_tempa, Hmem_size, cudaMemcpyDeviceToHost);
		printf("	P*H' = \n");
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < n; j++) {
				printf("	%f",h_tempe[i+j*2]);
			}
			printf("\n");
		}
		free(h_tempe);
	}

	//#####################################################################
	/* P*H'*inv(R)
	 * INPUT	P*H' is 2xN		inv(R) is NxN
	 * OUTPUT	P*H'*inv(R) is 2xN
	 * TODO If every measurement has the same sigma then it would be faster to simply multiply by the scalar
	 * cublasSetVector(2*n, sizeof(float), h_H, 1, d_tempf, 1);
	 * float Rinv = (1.0/R[0]);
	 * cublas_status = cublasSscal(cublas_handle, 2*n, &Rinv, d_tempf, 1);
	 */
	if (verbose) { 	printf("\nLaunching matrix multiplication using cuBLAS: 2x%i * %ix%i \n", n, n, n);
					fprintf(fpreport,"\nLaunching matrix multiplication using cuBLAS: 2x%i * %ix%i \n", n, n, n); }

	cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, n, n, alpha, d_tempa, 2, d_Rinv, n, beta, d_tempf, 2);
	assert(cublas_status == CUBLAS_STATUS_SUCCESS);

	if (DEBUG || vv) {	// check CUDA intermediate calculation
		float *h_tempf = (float *)malloc(Hmem_size);
		cudaMemcpy(h_tempf, d_tempf, Hmem_size, cudaMemcpyDeviceToHost);
		printf("	P*H'*inv(R) = \n");
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < n; j++) {
				printf("	%f",h_tempf[i+j*2]);
			}
			printf("\n");
		}
		free(h_tempf);
	}

	//#####################################################################
	/* z-h
	 * TODO this can happen simultaneously with something else... move to front after 'h' is calculated
	 * INPUT	z is Nx1	h is Nx1
	 * OUTPUT	z-h is Nx1
	 */
	if (verbose) { 	printf("\nCalculating measurement error: %ix1 - %ix1 \n", n, n);
					fprintf(fpreport,"\nCalculating measurement error: %ix1 - %ix1 \n", n, n); }
	cudaMemcpy(d_locy, z, mem_size, cudaMemcpyHostToDevice);	// re-use the d_locx and d_locx memory since we're done using it
	cudaEventRecord(start, 0);
	if(usesmem) {
		subtract<<< blocks, threads, 2*n*sizeof(float) >>>(d_locy, d_h, d_locx, n);
	} else {
		subtract<<< blocks, threads >>>(d_locy, d_h, d_locx, n);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	if (verbose) { 	printf("	Thread execution time: %f ms\n", elapsedTime);
					fprintf(fpreport,"	Thread execution time: %f ms\n", elapsedTime); }
	avgsub = avgsub + elapsedTime;
	if (elapsedTime < fastsub) {
		fastsub = elapsedTime;
		fastsubblock = blocks;
		fastsubthread = threads;
	}

	if (DEBUG || vv) {	// check CUDA intermediate calculation
		float *h_tempg = (float *)malloc(mem_size);
		cudaMemcpy(h_tempg, d_locx, mem_size, cudaMemcpyDeviceToHost);
		printf("	z-h =\n");
		for (int i = 0; i < n; i++) {
			printf("	%f\n",h_tempg[i]);
		}
		free(h_tempg);
	}

	//#####################################################################
	/* P*H'*inv(R)*(z-h)
	 * INPUT	P*H'*inv(R) is 2xN	(z-h) is Nx1
	 * OUTPUT	P*H'*inv(R)*(z-h) is 2x1
	 */
	if (verbose) { 	printf("\nLaunching matrix multiplication using cuBLAS: 2x%i * %ix1 \n", n, n);
					fprintf(fpreport,"\nLaunching matrix multiplication using cuBLAS: 2x%i * %ix1 \n", n, n); }

	cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 1, n, alpha, d_tempf, 2, d_locx, n, beta, d_temph, 2);
	cudaMemcpy(h_temph, d_temph, 2*sizeof(float), cudaMemcpyDeviceToHost);

	if (DEBUG || vv) {	// check CUDA intermediate calculation
		printf("	P*H'*inv(R)*(z-h) = \n");
		printf("	%f,  %f\n", h_temph[0], h_temph[1]);
	}

	//#####################################################################
	// xhat + P*H'*inv(R)*(z-h)
	if (verbose) { 	printf("\nUpdating estimated target location \n");
					fprintf(fpreport,"\nUpdating estimated target location \n"); }
	xhatx = xhatx + h_temph[0];
	xhaty = xhaty + h_temph[1];
	if (verbose) { 	printf("	xhat = %f, %f\n", xhatx, xhaty);
					fprintf(fpreport,"	xhat = %f, %f\n", xhatx, xhaty); }


	//TODO loop above, only pass here when done with geolocation calculation (error/count threshholds)

	//#####################################################################
	/* eig(P)
	 * TODO this can be moved up and done simultaneously (technically you should wait until xhat is what you want)
	 * Assume P is always 2x2
	 * TODO d_P is destroyed by this calculation, make sure that's ok for the rest of the code
	 * TODO Eigenvectors aren't quite identical to MATLAB, different signs and order, this might matter...
	 * This calculates work space needed for lwork but can be hardcoded since P should always be 2x2
	 * 		cusolver_status = cusolverDnSgesvd_bufferSize(cusolver_handle,2,2,&lwork );
	 * 		assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);
	 * 		printf("lwork size %i\n", lwork);
	 */
	if (verbose) { 	printf("\nCalculating eigenvalues of covariance matrix P using cuSOLVER: eig(2x2)\n");
					fprintf(fpreport,"\nCalculating eigenvalues of covariance matrix P using cuSOLVER: eig(2x2)\n"); }

	cusolver_status = cusolverDnSgesvd (cusolver_handle, jobu, jobvt, 2, 2, d_P, 2, d_Eig, d_tempb, 2, d_VT, 2, d_work, lwork, d_rwork, devInfo);
	cudaDeviceSynchronize();	//TODO why do I need this here?
	assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

	cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
	if (info_gpu != 0) {
		printf("Eigenvalue calculation unsuccessful, gesvd reports: info_gpu = %d\n", info_gpu);
	}

	if (DEBUG || vv) {
		cudaMemcpy(U , d_tempb , Pmem_size, cudaMemcpyDeviceToHost);	// re-use d_tempb for d_U
		cudaMemcpy(VT, d_VT, Pmem_size, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_Eig , d_Eig , sizeof(float)*2, cudaMemcpyDeviceToHost);
		printf("	eig(P) = \n");
		printf("	%f, %f\n",h_Eig[0],h_Eig[1]);
	    printf("	U = (matlab base-1)\n");
	    printf("	%f, %f\n",U[0],U[2]);
	    printf("	%f, %f\n",U[1],U[3]);
	    printf("	VT = (matlab base-1)\n");
	    printf("	%f, %f\n",VT[0],VT[2]);
	    printf("	%f, %f\n",VT[1],VT[3]);
	}

	//#####################################################################
	/* sqrt(k*max(eigenvalues)); sqrt(k*min(eigenvalues));
	 * TODO make this flexible so that we can calculate many versions of this simultaneously - 2xn
	 */
	if (verbose) { 	printf("\nCalculating 95%% semimajor, semiminor ellipse bounds \n");
					fprintf(fpreport,"\nCalculating 95%% semimajor, semiminor ellipse bounds \n");	}

	cudaEventRecord(start, 0);
	if (usesmem) {
		semiMajMinSmem<<< blocks, threads, 2*n*sizeof(float) >>>(d_Eig, k, d_tempi, num_ellipses);
		usesmem=0;	// TODO last use, set to zero so we can re-run and catch cmem or other kernel types on next run
	} else {
		semiMajMin<<< blocks, threads >>>(d_Eig, k, d_tempi, num_ellipses);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	if (verbose) { 	printf("	Thread execution time: %f ms\n", elapsedTime);
					fprintf(fpreport,"	Thread execution time: %f ms\n", elapsedTime); }
	avgsemi = avgsemi + elapsedTime;
	if (elapsedTime < fastsemi) {
		fastsemi = elapsedTime;
		fastsemiblock = blocks;
		fastsemithread = threads;
	}

	cudaMemcpy(h_semi, d_tempi, Emem_size, cudaMemcpyDeviceToHost);
	if (verbose) { 	printf("	semimajor = %f\n",h_semi[0]);			// semimajor is always first, output of calculation orders values based on magnitude
					printf("	semiminor = %f\n",h_semi[1]);
					printf("\n");
					fprintf(fpreport,"	semimajor = %f\n",h_semi[0]);
					fprintf(fpreport,"	semiminor = %f\n",h_semi[1]);
					fprintf(fpreport,"\n"); }

	//TODO loop here when calculating multiple geolocations
	/*  MATLAB output to check against for two DOA measurement values
	 *
	// starting data
	 	sigma = 2.0/180.0*M_PI;		// variance of measurements, assume single sensor

		ax.push_back((sin(60.0/180.0*M_PI)*15.0*1852.0));	// measurement location 1
		ay.push_back((cos(60.0/180.0*M_PI)*15.0*1852.0));
		z.push_back(-(90.0+60.0)/180.0*M_PI);				// measurement DOA 1

		ax.push_back((sin(-5.0/180.0*M_PI)*10.0*1852.0));	// measurement location 2
		ay.push_back((cos(5.0/180.0*M_PI)*10.0*1852.0));
		z.push_back(-(90.0-5.0)/180.0*M_PI);				// measurement DOA 2

		xhatx = 2.0*1852.0;				// target location guess
		xhaty = 2.0*1852.0;

	// output
		h = -2.6775
			-1.2246

		H =	0.000020	-0.000039
			0.000060	0.000022

		inv(R) =	820.7	0.0
					0.0		820.7

		H'*inv(R) =	0.0161	0.0493
					-0.0322	0.0178

		H'*inv(R)*H =	0.00003273	0.00000432
						0.00000432	0.00001651	

		P = 	316462 	-82777
				-82777 	627203

		P*H' =	9.4747		17.1997
				-26.2705	8.6074

		P*H'*inv(R) =	7776	14116
						-21560	7064

		P*H'*inv(R)*(z-h) = -3190.8
							-3113.7

		xhatnew = 	513.2420
					590.2777
		
		error = 3190.8
				3113.7	
	
		eigenvalues =	295787
						647878

		semimajor =	1970.2
		semiminor =	1331.2
	*/
	/*  MATLAB output to check against for three DOA measurement values

		// Add the following measurement to above
			ax.push_back((sin(10.0/180.0*M_PI)*5.0*1852.0));	// measurement location 3
			ay.push_back((cos(10.0/180.0*M_PI)*5.0*1852.0));
			z.push_back(-(90.0+10.0)/180.0*M_PI);				// measurement DOA 3

		// output
			h = -2.6775
				-1.2246
				-1.2015

			H =	0.000020	-0.000039
				0.000060	0.000022
				0.000160	0.000062

			inv(R) =	820.7	0.0		0.0
						0.0		820.7	0.0
						0.0		0.0		820.7

			H'*inv(R) =	0.0161	0.0493	0.1318
						-0.0322	0.0178	0.0510

			H'*inv(R)*H =	0.000024	0.000009
							0.000009	0.000005

			P = 	110925.37 	-198389.04
					-198389.04 	562173.68

			P*H' =	9.9757		2.3629	5.4826
					-25.9886	0.2619	3.0839

			P*H'*inv(R) =	8187.14		1939.23	4499.62
							-21328.96	214.96	2530.96

			P*H'*inv(R)*(z-h) = -2461.075
								-2703.292

			xhatnew = 	1242.92
						1000.70

			error = 3190.8
					3113.7

			eigenvalues =	636990.00
							36109.05

			semimajor =	1953.59
			semiminor =	465.13
		*/

	// release memory
	free(h_locx);
	free(h_locy);
	free(h_Rinv);
	free(alpha);
	free(beta);
	free(h_temph);

	cudaFree(d_locx);
	cudaFree(d_locy);
	cudaFree(d_h);
	cudaFree(d_tempa);
	cudaFree(d_H);		
	cudaFree(d_Rinv);
	cudaFree(d_P);
	cudaFree(d_tempb);
	cudaFree(P);
	cudaFree(INFO);
	cudaFree(d_tempc);
	cudaFree(d_tempd);
	cudaFree(d_tempf);
	cudaFree(d_temph);
	if (d_VT   ) cudaFree(d_VT);
	if (devInfo) cudaFree(devInfo);
	if (d_work ) cudaFree(d_work);
	if (d_Eig  ) cudaFree(d_Eig);
	if (d_tempi) cudaFree(d_tempi);

	if (cublas_handle) cublasDestroy_v2(cublas_handle);
	if (cusolver_handle) cusolverDnDestroy(cusolver_handle);
	cudaDeviceReset();


	clock_gettime(CLOCK_REALTIME, &tp2);
	time_t s2;
	unsigned long us2;
	if (tp2.tv_nsec < tp1.tv_nsec) {
		s2 = (tp2.tv_sec-tp1.tv_sec) - 1;
		us2 = (tp2.tv_nsec + (1000000000 - tp1.tv_nsec)) / 1000;
	} else {
		s2 = tp2.tv_sec-tp1.tv_sec;
		us2 = (tp2.tv_nsec-tp1.tv_nsec) / 1000;
	}

	avgcuexecs = avgcuexecs + s2;
	avgcuexecus = avgcuexecus + us2;
	if (s2 < fastcuexecs || (s2 == fastcuexecs && us2 < fastcuexecus)) {
		fastcuexecs = s2;
		fastcuexecus = us2;
		fastcuexecblock = blocks;
		fastcuexecthread = threads;
	} else if (s2 > worstcuexecs || (s2 == worstcuexecs && us2 > fastcuexecus)) {
		worstcuexecs = s2;
		worstcuexecus = us2;
		worstcuexecblock = blocks;
		worstcuexecthread = threads;
	}

	if (true) { 	printf("CUDA Geolocation execution time: %i s  %lu us\n", (int)s2, us2);
					fprintf(fpreport,"CUDA Geolocation execution time: %i s  %lu us\n", (int)s2, us2);
					printf("\n");
					fprintf(fpreport,"\n");
					fflush(fpreport); }

	// save geolocations in csv file
	geofile = fopen("georesults.csv", "w");
	fprintf(geofile, "%f,%f,%f,%f\n", xhatx, xhaty, h_semi[0], h_semi[1]);
	if (fclose(geofile) != 0) {
		fprintf(stderr,"Failed to close data file %s with error %d\n", "georesults.csv", errno);
		exit(1);
	}
	if (h_semi)  free(h_semi);

	fprintf(benchmarkfile, "%i,%lu,%i,%lu,", (int)s1, us1, (int)s2, us2);
}


// TODO add data association, multiple targets

int main(int argc, char **argv) {

	fpreport = fopen("report.txt", "w");
	if (fpreport==NULL) {
		fprintf(stderr,"Failed to open file %s with error %d \n", "report.txt", errno);
		exit(1);
	}

	printf("##### Starting cuGEO! #####\n\n");
	fprintf(fpreport,"##### Starting cuGEO! #####\n\n");

	checkCmdArgs(argc, argv);

	checkDeviceProperties();

	benchmarkfile = fopen("benchmark.csv", "w");
	fprintf(benchmarkfile, "C/C++ execute s, C/C++ execute us, CUDA memcpy s, CUDA memcpy us, CUDA execute s, CUDA execute us, threads, blocks\n");
	int final_measurements = measurements;
	for (measurements = 3; measurements <= final_measurements; measurements = measurements*2) { 		// TODO might want to re-arrange these nested loops

		if (!benchmark) {						// makes sure that people WANT to run all the iterations, otherwise only do the final one (requested measurements)
			measurements = final_measurements;
		}

		float *zgen = generateScenario();

		cpuGeolocation(zgen);

		//TODO iterate on Geolocation for different target location guesses
		/*int count = 1;
		int maxcount = 100;
		float error = 100.0;
		float maxerror = 0.1;*/

		// re-generate scenario, don't want to have any values left-over from C/C++ test
		zgen = generateScenario();

		printf("\n##### Executing CUDA Geo Benchmarking Routine #####\n");
		fprintf(fpreport,"\n##### Executing CUDA Geo Benchmarking Routine #####\n");
		int minblocks = 1;
		int maxblocks = measurements; // there is effectively no max (my GPU is 2147483647 in dim[0]) so max is 1 thread per block

		// calculate min number of blocks based on GPU and number of measurements
		if (measurements%maxThreadsPerBlock == 0) {
			minblocks = measurements / maxThreadsPerBlock;
		} else {
			printf("Caution, number of measurements is not evenly divisible by maxThreadsPerBlock, could have issues...\n");
			fprintf(fpreport,"Caution, number of measurements is not evenly divisible by maxThreadsPerBlock, could have issues...\n");
			minblocks = measurements / maxThreadsPerBlock + 1;
		}

		// TODO for now we'll reduce thread count per block by factor of 2 each iteration (or increase block count by factor of 2)
		//for (blocks = minblocks; blocks < maxblocks; blocks = blocks*2) {

		if (!blocksflag && !threadsflag) {	// If blocksflag wasn't set, then use smart block/thread count
			blocks = minblocks;
			threads = measurements / blocks;
		} else if (blocksflag && !threadsflag) {
			threads = measurements / blocks;
		} else if (threadsflag && !blocksflag) {
			blocks = measurements / threads;
		} else {
			// if both blocks and threads were set by user, then let 'em have it
			if (threads*blocks != measurements) {
				fprintf(stderr,"Error, chosen parameters are nonsense:	%i threads * %i blocks != %i measurements\n", threads, blocks, measurements);
				fprintf(stderr, "Try NOT picking specific thread/block counts. I can figure it out from the number of measurements.");
				exit(1);
			}
		}

		printf(" 		blocks used = %i\n", blocks);
		printf(" 		threads used = %i\n", threads);

		/*if (measurements%blocks != 0) {
			printf("Something went wrong, data is not evenly divisible by # of blocks...\n");
			fprintf(fpreport,"Something went wrong, data is not evenly divisible by # of blocks...\n");
		}*/

		for (int j = 0; j < (usesmem+usecmem+1); j++) {		// ex. you want to check smem and cmem types so 2 + default = 3 runs
			// tests should execute this memory type order regardless of which chosen by user
			char *typestring;
			if (usesmem) { typestring = "shared memory"; } else if (usecmem) {typestring = "constant memory"; } else { typestring = "mixed memory"; }

			for (int i = 0; i < iterations; i++) {
				if (verbose) { 	printf("## Testing %d blocks of %d threads ##\n", blocks, threads);
								fprintf(fpreport,"## Testing %d blocks of %d threads ##\n", blocks, threads);
								fflush(fpreport); }
				cudaGeolocation(zgen);
			}

			fprintf(benchmarkfile, "%i,%i,%i\n", threads, blocks, measurements);

			// compute averages
			// TODO remember to check how the average is done when multiple iterations of the same block/thread are run vice multiple iterations with different thread/blocks
			avgparam = avgparam / iterations;
			avgcov = avgcov / iterations;
			avgsub = avgsub / iterations;
			avgsemi = avgsemi / iterations;
			avgcuexecs = avgcuexecs / iterations;

			if (verbose) {
				printf("\n##### Completed CUDA Geo Benchmarking Routine #####\n");
				fprintf(fpreport,"\n##### Completed CUDA Geo Benchmarking Routine #####\n");

				printf("Memory used:		%s\n", typestring);
				printf("Num blocks:		%i\n", blocks);
				printf("Num threads:		%i\n", threads);
				printf("Iterations:		%i\n", iterations);
				printf("Num measurements:	%i\n", measurements);
				printf("\n");
				fprintf(fpreport,"Memory used:		%s\n", typestring);
				fprintf(fpreport,"Blocks used:		%i\n", blocks);
				fprintf(fpreport,"Threads used:		%i\n", threads);
				fprintf(fpreport,"Iterations:		%i\n", iterations);
				fprintf(fpreport,"Num measurements:	%i\n", measurements);
				fprintf(fpreport,"\n");

				// print fastest and average execution times based on layout
				printf("Fastest execution time for entire C/C++ Geo-location algorithm: %i s, %lu us\n", fastcexecs, fastcexecus);
				printf("Slowest execution time for entire C/C++ Geo-location algorithm: %i s, %lu us\n", worstcexecs, worstcexecus);
				printf("Average execution time for entire C/C++ Geo-location algorithm: %i s, %lu us\n", avgcexecs, avgcexecus);
				printf("\n");
				fprintf(fpreport,"Fastest execution time for entire C/C++ Geo-location algorithm: %i s, %lu us\n", fastcexecs, fastcexecus);
				fprintf(fpreport,"Slowest execution time for entire C/C++ Geo-location algorithm: %i s, %lu us\n", worstcexecs, worstcexecus);
				fprintf(fpreport,"Average execution time for entire C/C++ Geo-location algorithm: %i s, %lu us\n", avgcexecs, avgcexecus);
				fprintf(fpreport,"\n");
				printf("Fastest execution time for entire CUDA Geo-location algorithm: %i s, %lu us using %i blocks and %i threads\n", fastcuexecs, fastcuexecus, fastcuexecblock, fastcuexecthread);
				printf("Slowest execution time for entire CUDA Geo-location algorithm: %i s, %lu us using %i blocks and %i threads\n", worstcuexecs, worstcuexecus, worstcuexecblock, worstcuexecthread);
				printf("Average execution time for entire CUDA Geo-location algorithm: %i s, %lu us\n", avgcuexecs, avgcuexecus);
				printf("\n");
				fprintf(fpreport,"Fastest execution time for entire CUDA Geo-location algorithm: %i s, %lu us using %i blocks and %i threads\n", fastcuexecs, fastcuexecus, fastcuexecblock, fastcuexecthread);
				fprintf(fpreport,"Slowest execution time for entire CUDA Geo-location algorithm: %i s, %lu us using %i blocks and %i threads\n", worstcuexecs, worstcuexecus, worstcuexecblock, worstcuexecthread);
				fprintf(fpreport,"Average execution time for entire CUDA Geo-location algorithm: %i s, %lu us\n", avgcuexecs, avgcuexecus);
				fprintf(fpreport,"\n");
				printf("Fastest parameterPrediction kernel execution: %f ms using %i blocks and %i threads\n", fastparam, fastparamblock, fastparamthread);
				printf("Average parameterPrediction kernel execution: %f ms\n", avgparam);
				printf("\n");
				fprintf(fpreport,"Fastest parameterPrediction kernel execution: %f ms using %i blocks and %i threads\n", fastparam, fastparamblock, fastparamthread);
				fprintf(fpreport,"Average parameterPrediction kernel execution: %f ms\n", avgparam);
				fprintf(fpreport,"\n");
				printf("Fastest covariancePrediction kernel execution: %f ms using %i blocks and %i threads\n", fastcov, fastcovblock, fastcovthread);
				printf("Average covariancePrediction kernel execution: %f ms\n", avgcov);
				printf("\n");
				fprintf(fpreport,"Fastest covariancePrediction kernel execution: %f ms using %i blocks and %i threads\n", fastcov, fastcovblock, fastcovthread);
				fprintf(fpreport,"Average covariancePrediction kernel execution: %f ms\n", avgcov);
				fprintf(fpreport,"\n");
				printf("Fastest subtract kernel execution: %f ms using %i blocks and %i threads\n", fastsub, fastsubblock, fastsubthread);
				printf("Average subtract kernel execution: %f ms\n", avgsub);
				printf("\n");
				fprintf(fpreport,"Fastest subtract kernel execution: %f ms using %i blocks and %i threads\n", fastsub, fastsubblock, fastsubthread);
				fprintf(fpreport,"Average subtract kernel execution: %f ms\n", avgsub);
				fprintf(fpreport,"\n");
				printf("Fastest semiMajMin kernel execution: %f ms using %i blocks and %i threads\n", fastsemi, fastsemiblock, fastsemithread);
				printf("Average semiMajMin kernel execution: %f ms\n", avgsemi);
				printf("\n");
				fprintf(fpreport,"Fastest semiMajMin kernel execution: %f ms using %i blocks and %i threads\n", fastsemi, fastsemiblock, fastsemithread);
				fprintf(fpreport,"Average semiMajMin kernel execution: %f ms\n", avgsemi);
				fprintf(fpreport,"\n");
			}
		}
	}

	if (fclose(benchmarkfile) != 0) {
		fprintf(stderr,"Failed to close data file %s with error %d\n", "benchmark.csv", errno);
		exit(1);
	}

	cudaDeviceReset();
	printf("\nDone\n");
	fprintf(fpreport,"\nDone\n");
	if (fclose(fpreport) != 0) {
		fprintf(stderr,"Failed to close data file %s with error %d\n", "report.txt", errno);
		exit(1);
	}

	// execute Python plot script
	if(plot) {
		Py_Initialize();
		FILE* pyfile = fopen("plot.py", "r");
		PyRun_SimpleFileEx(pyfile, "plot.py", 1);
		Py_Finalize();
	}

	exit(0);
}

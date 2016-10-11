#### cuGEO ####
cuGEO is a CUDA implementation of a Geo-location algorithm using linear/non-linear estimation techniques including Iterated Least Squares (ILS) and Kalman Filters. 

To build the project:
	
	nvcc cuGEO.cu kernels.cu -o cuGEO -lcublas -lcusolver -llapacke -llapack -lblas

To run the program with default settings and all printed output:

	./cuGEO --verbose

This will use a single target location and three Direction of Arrival (DOA) measurements with an initial guess to estimate the target location using ILS. Other flags include '--config' which will read a configuration file for flag input, '--measurements' which takes an integer input for the number of DOA measurements to generate/use.


#### TODO ####

	* add argument'--file' which reads a comma delimited file for one measurement per line in the format "<locationx>,<locationy>,<DOA>,<sigma>\n". When this input is used, the number of measurements in the file will be the total number of measurements processed by the geo-location algorithm, unless the '--measurements' arg is set which will override the total (up to the file size but not more).

	* finish C/C++ implementation
	* fix block/thread user choice
	* improve matrix inversion routine (or use alternate means)
	* loop ILS for a count/error amount 

	* add loop for different target guesses, allow user to input guess (hard coded real target location)
	* add a function to auto-generate scenarios with valid target/measurement values
	* allow user to input real target location, guess, flight path, etc.
	* add multiple target, data association methods
	* add plotting features



#### Nvidia, Docker, and GitLab CI ####

Check your nvidia driver version
	sudo nvidia-settings -v

Create a named volume based on that version
	sudo docker volume create --name=nvidia_driver_367.27 -d nvidia-docker

Try it out
	sudo docker run -ti --rm  --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia0 --volume=nvidia_driver_367.27:/usr/local/nvidia:ro tc/cuda:7.5-devel-centos7

Using a different terminal get the container ID
	sudo docker ps

Copy in an exectuable and try it
	sudo docker cp <executable> <containerID>:/

If that works, try to update the GitLab runner
	sudo vi /etc/gitlab-runner/config.toml

Add these fields to your runner under  [runners.docker]
	devices = ["/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia0"]
	volumes = ["/cache","nvidia_driver_367.27:/usr/local/nvidia:ro"]
	
Restart the runners
	sudo gitlab-runner restart

Cross your fingers....



This project will be tagged for class EN-605-417 based on module deliveries which will contribute to overall project completion.

#### Module 1 - 3 ####
Basic CUDA test harness with simple calculations and command line options.


#### Module 4 ####
Adds  more complicated calculations and grid/block decisions. The initial stages of a Geo-location algorithm were added, specifically the parameter state and covariance prediction steps of a Kalman Filter for Direction of Arrival (DOA) measurements. These calculations build on the initial arctan kernel added in Module 1-3 to include sin, cos, and pow functions all combined to calculate the covariance prediction of the DOA measurement. For now, the measurements are auto-generated and do not have a real-world meaning, but I plan to create a data set for testing the overall geo-location algorithm that can be imported from a file.


#### Module 5 ####
Adds utilization of GPU registers and a test report. Register count is determined at compile time, so the "maxrregcount" compiler flag is used to increase the number used to the max per thread. Also the only way to view the number of registers being used per kernel is to use additional compiler flags "-Xptxas=-v" which will print out memory allocations of each kernel.
	
	nvcc -maxrregcount 63 -Xptxas=-v cuGEO.cu kernels.cu -o cuGEO -lcublas -lcusolver -llapacke -llapack -lblas

A test report is now produced "report.txt" for every run of the program that contains the same output that is sent to the terminal. The CI script was improved so that the same terminal/report output is printed in the CI test window, and the report is saved as an artifact.


#### Module 6 ####
Adds utilization of GPU cache based on certain parameters. The CUDA libraries cuBLAS, cuSOLVER and the C libraries BLAS, LAPACK are now included to perform the matrix operations needed by the geo-location algorithm. Functions for matrix multiplication and inversion are used and compared for both the host (C/C++) and device (CUDA) code.


#### Module 7 ####
Adds utilization of GPU constant memory.


#### Module 8 ####
Adds utilization of GPU global memory.


#### Module 10 ####
Adds utilization of multiple GPU's.



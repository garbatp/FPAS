
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"
#include "math_functions.h"
#include "math_constants.h":
#include <cufft.h>

#include <stdio.h>
#include "common\cpu_bitmap.h"
#include "common\book.h"

struct DataBlock {
	unsigned char   *dev_bitmap;
};


struct kernelConf
{
	dim3 block;
	dim3 grid;
};

kernelConf* conf_FFT_Shift(int N, int batch)
{
	kernelConf* conf = (kernelConf*)malloc(sizeof(kernelConf));

	int threadsPerBlock_X;

	threadsPerBlock_X = 1024;

	conf->block = dim3(threadsPerBlock_X, 1, 1);
	conf->grid = dim3(((N*batch / threadsPerBlock_X)) + 1, 1, 1);

	return conf;
}

__global__ void cufftShift_2D(cufftComplex* data, int N, int batch)
{

	int sLine = N;
	int sSlice = N * N;


	int sEq1 = (sSlice + sLine) / 2;
	int sEq2 = (sSlice - sLine) / 2;

	int threadIdxX = threadIdx.x;
	int blockDimX = blockDim.x;
	int blockIdxX = blockIdx.x;

	cufftComplex regTemp;
	int index = ((blockIdxX * blockDimX) + threadIdxX);
	int batchNumber = index / (N*N);

	int yIndex = (index / N) - batchNumber*N;
	int xIndex = index - (N*(index / N));


	if (batchNumber <= (batch - 1) && xIndex < N / 2)
	{
		if (batchNumber <= (batch - 1) && yIndex < N / 2)
		{
			regTemp = data[index];


			data[index] = data[index + sEq1];


			data[index + sEq1] = regTemp;
		}
	}
	else
	{
		if (batchNumber <= (batch - 1) && yIndex < N / 2)
		{
			regTemp = data[index];

			data[index] = data[index + sEq2];


			data[index + sEq2] = regTemp;
		}
	}


}

__global__ void copy2bitmap(cuComplex *ins, unsigned char *ptr) {
	// Odwzorowanie z blockIdx na po³o¿enie piksela
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	double cccc = ins[offset].x;

	float aaa = (atan2(ins[offset].y, ins[offset].x));
	
	ptr[offset * 4 + 0] = 255 * abs(aaa) / 3.14 ;//2550000 * abs(ins[offset].x);//(atan2(in[offset].y, in[offset].x)); //
	ptr[offset * 4 + 1] = 255 * abs(aaa) / 3.14;//2550000 * abs(ins[offset].y);//;
	ptr[offset * 4 + 2] = 255* abs(aaa)/3.14;
	ptr[offset * 4 + 3] = 255;
}



__global__ void shift2Dout(cuComplex *input, cuComplex *output)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	int n = blockIdx.x;
	int m = blockIdx.y;
	int di = blockDim.x / 2;
	int dj = blockDim.y / 2;

	output[i + n*blockDim.x + j*blockDim.x*gridDim.x + m*blockDim.x*blockDim.y*gridDim.x] = input[i + j*blockDim.x + n*blockDim.x*blockDim.y + m*blockDim.x*blockDim.y*gridDim.x];

}


__device__ __forceinline__ cuComplex expf(cuComplex z)
{

	cuComplex res;
	float t = expf(z.x);
	sincosf(z.y, &res.y, &res.x);
	res.x *= t;
	res.y *= t;

	return res;

}

__global__ void calculate(cuComplex *fths, int *xo, int *yo, double *uo, double *zo2, float dfxs, float lambda, float k0, int Ts, float *fxs, float * y0seg, float* x0seg, int S_Bx, int S_By, int N_Bx, int N_By, int q)
{



	float yp = yo[threadIdx.x] - y0seg[blockIdx.y];

	float xp = xo[threadIdx.x] - x0seg[blockIdx.x];

    double rp = sqrt(zo2[threadIdx.x] + xp*xp + yp*yp);


    float inv_rp = 1 / rp;

    float fxp = xp*inv_rp / lambda;
	float fyp = yp*inv_rp / lambda;

cuComplex res;
	cuComplex c0;
	cuComplex arg;
	cuComplex arg1;
	double eps = 2.2204e-16;

	int iifx = round(fxp / dfxs) + S_Bx / 2 ;
	int iify = round(fyp / dfxs) + S_By / 2 ;

	if (iifx <= 0 || iifx >= S_Bx || iify <= 0 || iify >= S_Bx){
		iifx = S_Bx / 2 ;
		iify = S_Bx / 2 ;
		arg.x = eps;

		sincosf(arg.x, &c0.y, &c0.x);
		c0.x *= eps;
		c0.y *= eps;
	/*	c0.x = 1;
		c0.y = 2;*/
	}
	else
	{
		
		arg.x = (k0*rp - 2 * CUDART_PI_F*(fxs[iifx] + fxs[iify])*(Ts / 2));
		arg1.x = (2 * CUDART_PI_F  * uo[threadIdx.x] / 6400);

		//	arg1.x = 2;
		//	arg.x = -35.699;
		
		float t = arg1.x*inv_rp;
		sincosf(-arg.x, &res.y, &res.x);
		res.x *= t;
		res.y *= t;

		//	c0 = expf(arg);
		//	cuComplex uoo = expf(arg1);
		//	c0.x = uo[threadIdx.x] * c0.x;
		//	c0.y = uo[threadIdx.x] * c0.y;
		c0 = res;
	/*	
		iifx = S_Bx / 2 ;
		iify = S_Bx / 2 ;
		c0.x = 1;
		c0.y = 2;*/

	}
	

//	fths[blockIdx.y + blockIdx.x*S_Bx + iifx*S_Bx*N_Bx + iify*S_Bx*N_Bx*S_By].x += c0.x;
//	fths[blockIdx.y + blockIdx.x*S_Bx + iifx*S_Bx*N_Bx + iify*S_Bx*N_Bx*S_By].y += c0.y;
//	fths[iifx + blockIdx.x*S_Bx + iify*S_Bx*N_Bx + blockIdx.y* S_Bx*N_Bx*S_By].x += c0.x;
//	fths[iifx + blockIdx.x*S_Bx + iify*S_Bx*N_Bx + blockIdx.y* S_Bx*N_Bx*S_By].y += c0.y;

//	fths[iifx + iify*S_Bx + blockIdx.y*S_Bx*S_By + blockIdx.x* S_Bx*N_Bx*S_By].x += c0.x;
//	fths[iifx + iify*S_Bx + blockIdx.y*S_Bx*S_By + blockIdx.x* S_Bx*N_Bx*S_By].y += c0.y;

	fths[iifx + iify*S_Bx + blockIdx.x*S_Bx*S_By + blockIdx.y* S_Bx*N_Bx*S_By].x += c0.x;
	fths[iifx + iify*S_Bx + blockIdx.x*S_Bx*S_By + blockIdx.y* S_Bx*N_Bx*S_By].y += c0.y;
//	fths[iifx + iify*S_Bx + blockIdx.x*S_Bx*S_By + blockIdx.y* S_Bx*N_Bx*S_By].x = 128;
//	fths[iifx + iify*S_Bx + blockIdx.x*S_Bx*S_By + blockIdx.y* S_Bx*N_Bx*S_By].y = 128 ;

}


cufftResult preparePlan2D(cufftHandle* plan, int nRows, int nCols, int batch){

	int n[2] = { nRows, nCols };

	cufftResult result = cufftPlanMany(plan,
		2, //rank
		n, //dimensions = {nRows, nCols}
		0, //inembed
		batch, //istride
		1, //idist
		0, //onembed
		batch, //ostride
		1, //odist
		CUFFT_C2C, //cufftType
		batch /*batch*/);

	if (result != 0){

		//		std::cout << "preparePlan2D error, result: " << result << "/n";
		return result;
	}
	return result;
}

cufftResult execute2D(cufftHandle* plan, cufftComplex* idata, cufftComplex* odata, int direction){

	cufftResult result = cufftExecC2C(*plan, idata, odata, direction);

	if (result != 0){

		//		cout << "execute2D error, result: " << result << "/n";
		return result;
	}
	return result;
}




void FPAS_CGH_2D(int Np, int* xo, int* yo, double* zo, double* uo, int Nx, int Ny, int dx, float lambda, int S_Bx, int S_By, int q, cuComplex* fths_p, cuComplex* fths_s)
{
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	double k0 = 2 * CUDART_PI_F / lambda;

	int x_size = (Nx / 2) + ((Nx / 2) - 1) + 1;
	int y_size = (Ny / 2) + ((Ny / 2) - 1) + 1;

	float *x = (float*)malloc(x_size * sizeof(float));
	float *y = (float*)malloc(y_size * sizeof(float));

	for (int t = 0; t < x_size; t++){
		x[t] = (-Nx / 2 + t)*dx;
	}

	for (int t = 0; t < y_size; t++){
		y[t] = (-Ny / 2 + t)*dx;
	}

	int N_Bx = Nx / S_Bx; // dodaæ obs³ugê nie ca³kowitych dzieleñ
	int N_By = Ny / S_By;

	int Ts = S_Bx*dx;

	float dfxs = 1 / (float)Ts;

	int fxs_size = (S_Bx / 2) + ((S_Bx / 2) - 1) + 1;
	float *fxs = (float*)malloc(fxs_size * sizeof(float));

	for (int t = 0; t < fxs_size; t++){
		fxs[t] = (float)(-S_Bx / 2 + t)*dfxs;
	}

	float * x0seg = (float*)malloc((N_Bx)* sizeof(float));

	for (int t = 0; t < N_By; t++){
		x0seg[t] = (x[0] + (t*Ts) + Ts / 2);
	}

	float * y0seg = (float*)malloc((N_By)* sizeof(float));

	for (int t = 0; t < N_By; t++){
		y0seg[t] = (y[0] + (t*Ts) + Ts / 2);
	}


	long double *z02;
	z02 = (long double*)malloc((Np)* sizeof(long double));

	for (int t = 0; t < Np; t++)
		z02[t] = zo[t] * zo[t];

	z02[0];


	int *d_xo;
	int *d_yo;
	 double *d_z02;
	double *d_uo;

	cudaMalloc((void**)&d_xo, sizeof(int)*Np);
	cudaMalloc((void**)&d_yo, sizeof(int)*Np);
	cudaMalloc((void**)&d_z02, sizeof( double)*Np);
	cudaMalloc((void**)&d_uo, sizeof(double)*Np);

	cudaMemcpy(d_xo, xo, Np*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_yo, yo, Np*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_z02, z02, Np*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_uo, uo, Np*sizeof(double), cudaMemcpyHostToDevice);

    float *d_fxs;
	float *d_y0seg;
	float *d_x0seg;

	cudaMalloc((void**)&d_x0seg, sizeof(float)*N_Bx);
	cudaMalloc((void**)&d_y0seg, sizeof(float)*N_By);
	cudaMalloc((void**)&d_fxs, sizeof(float)*fxs_size);

	cudaMemcpy(d_fxs, fxs, fxs_size*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x0seg, x0seg, N_Bx*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y0seg, y0seg, N_By*sizeof(float), cudaMemcpyHostToDevice);

	dim3 grid;
	grid.x = N_Bx;//y
	grid.y = N_By;//x

	dim3 block;
	block.x = Np; //z
	block.y = 1;

	cudaEventRecord(start, 0);
	calculate << < grid, block >> >(fths_p, d_xo, d_yo, d_uo, d_z02, dfxs, lambda, k0, Ts, d_fxs, d_y0seg, d_x0seg, S_Bx, S_Bx, N_Bx, N_By, q);

	cufftComplex* h_out; //dane wynikowe CPU

	h_out = (cufftComplex*)malloc(sizeof(cufftComplex)*S_Bx*S_By*N_Bx*N_By); //allokacja pamiêci na wynik (CPU)

	cudaMemcpy(h_out, fths_p, sizeof(cufftComplex)*S_Bx*S_By*N_Bx*N_By, cudaMemcpyDeviceToHost);
	for (int iii = 0; iii < Nx*Ny; iii++)
	{
		if (h_out[iii].x != 0)
			printf("T: %f + i%f\n", 10e15*h_out[iii].x, 10e15*h_out[iii].y);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time for the kernel: %f ms\n", time);



//	dim3 grid;
	grid.x = N_Bx;
	grid.y = N_By;

//	dim3 block;
	block.x = S_Bx;
	block.y = S_By;

	cudaEventRecord(start, 0);
	shift2Dout << < grid, block >> > (fths_p, fths_s);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("Time for the kernel: %f ms\n", time);

	/*	cuComplex *host;
	host = (cuComplex*)malloc(sizeof(cuComplex)*Nosx*Nosy*Np);
	cudaMemcpy(host, fths, sizeof(cuComplex)*Nosx*Nosy*Np, cudaMemcpyDeviceToHost);
	*/
}

int main()
{

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/*START CUDA CALC PART - DEKLARACJE*/
	int Nx = 512;
	int Ny = 512;
	int dx = 8;
	float lambda = 0.5;
	/*START CUDA FFT 2D PART - DEKLARACJE*/
	int S_Bx = 16;
	int S_By = 16;

	//cufftComplex* h_out; //dane wynikowe CPU
	cufftComplex* holo; //dane wyjœciowe GPU

	int batch = Nx / S_Bx * Ny / S_By;  //N_Bx*N_By
	cufftHandle forwardPlan;

	preparePlan2D(&forwardPlan, S_Bx, S_By, batch);


	//h_out = (cufftComplex*)malloc(sizeof(cufftComplex)*S_Bx*S_By*batch); //allokacja pamiêci na wynik (CPU)

	cudaMalloc(&holo, sizeof(cufftComplex) *S_Bx*S_By*batch); //allokacja pamiêci na dane wyjœciowe (GPU)
	cudaMemset(holo, 0, sizeof(cufftComplex)*S_Bx*S_By*batch); //Wype³nianie zaalokowanej pamiêci zerami (GPU)

	/*END CUDA FFT 2D PART - DEKLARACJE*/

	/*Kod kernela*/
	int Np = 1024;

	int *xo;
	int *yo;
	double *zo;
	double *uo;


	xo = (int*)malloc((Np)* sizeof(int));
	yo = (int*)malloc((Np)* sizeof(int));
	zo = (double*)malloc((Np)* sizeof(double));
	uo = (double*)malloc((Np)* sizeof(double));

	for (int tt = 0; tt < Np; tt++)
	{
		xo[tt] = tt;
		yo[tt] = tt;
		zo[tt] = tt;
	}



	double W = 0.1e3;
	double dxo = W / 10.0;
	int foo = 0;

	for (foo = 0; foo < Np; foo++)
	{
	xo[foo] = W;
	yo[foo] = W;
	zo[foo] = 50e3; // 5e3;
	uo[foo] = 3.14;
	}
//	uo = exp(2 * pi * 1i * rand(1, Np) / 6400); % object point phase - random




	cuComplex *fths_p;  
	cuComplex *fths_s;

	cufftComplex* fhs;

	//	cudaMalloc(&fhs, sizeof(cufftComplex)*S_Bx*S_By*batch); //allokacja pamiêci na dane wejœciowe (GPU)
	cudaMalloc(&fths_p, sizeof(cuComplex)*Nx*Ny);
	cudaMemset(fths_p, 0, sizeof(cuComplex)*Nx*Ny);

	cudaMalloc(&fths_s, sizeof(cuComplex)*Nx*Ny);
	cudaMemset(fths_s, 0, sizeof(cuComplex)*Nx*Ny);

	cudaEventRecord(start, 0);
	/*START CUDA CALC PART */
	FPAS_CGH_2D(Np, xo, yo, zo, uo, Nx, Ny, dx, lambda, S_Bx, S_By, 2, fths_p, fths_s);



	/*START CUDA FFT_SHIFT PART */
	kernelConf * conf = conf_FFT_Shift(S_Bx*S_By, batch);

	cufftShift_2D << <conf->grid, conf->block >> >(fths_p, S_Bx, batch);
	/*END CUDA FFT_SHIFT PART */

	//cudaMemcpy(h_out, fths_p, sizeof(cufftComplex)*S_Bx*S_By*batch, cudaMemcpyDeviceToHost);
	//	for (int iii = 0; iii < Nx*Ny; iii++)
	//	printf("T: %f\n", h_out[iii]);


	/*START CUDA FFT PART */
	execute2D(&forwardPlan, fths_p, holo, CUFFT_FORWARD);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	/*Wyswietlanie modulu/fazy*/
//	cudaMemcpy(h_out, holo, sizeof(cufftComplex)*S_Bx*S_By*batch, cudaMemcpyDeviceToHost);

	dim3 grid;
	grid.x = Nx / S_Bx;
	grid.y = Ny / S_By;

	dim3 block;
	block.x = S_Bx;
	block.y = S_By;


	/*Wyswietlanie modulu/fazy*/
	//	cudaMemcpy(h_out, holo, sizeof(cufftComplex)*S_Bx*S_By*batch, cudaMemcpyDeviceToHost);

	shift2Dout << < grid, block >> > (holo, holo_f);
	/*END CUDA FFT PART */

	// Retrieve result from device and store it in host array
	cudaEventElapsedTime(&time, start, stop);
	printf("Time for the kernel: %f ms\n", time);
//	printf("Time for the kernel: %f ms\n", h_out[213100].x);

	printf("END \n");


		DataBlock   data;
		CPUBitmap bitmap(Nx, Ny, &data);
		unsigned char    *dev_bitmap;

		HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
		data.dev_bitmap = dev_bitmap;
		cudaMemset(dev_bitmap, 255, bitmap.image_size());

		dim3    grid(Nx, Ny);
		copy2bitmap << <grid, 1 >> >(holo, dev_bitmap);

		HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
			bitmap.image_size(),
			cudaMemcpyDeviceToHost));

		bitmap.display_and_exit();
	
	return 0;
}

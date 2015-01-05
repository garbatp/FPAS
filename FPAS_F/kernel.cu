
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



__global__ void copy2bitmap(cuComplex *in, unsigned char *ptr) {
	// Odwzorowanie z blockIdx na po³o¿enie piksela
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	float aaa = (atan2(in[offset].y, in[offset].x));
	
	ptr[offset * 4 + 0] = 0; //2550000 * abs(in[offset].x);//(atan2(in[offset].y, in[offset].x)); //
	ptr[offset * 4 + 1] = 0;//2550000 * abs(in[offset].y);//;
	ptr[offset * 4 + 2] = 255* abs(aaa);
	ptr[offset * 4 + 3] = 255;
}



__global__ void shift2Dout(cuComplex *input, cufftComplex *output)
{
    int i = threadIdx.x;
	int j = threadIdx.y;
	int n = blockIdx.x;
	int m = blockIdx.y;
	int di = blockDim.x / 2;
	int dj = blockDim.y / 2;

//	float *temp;
//	cudaMalloc(temp, sizeof(float)*blockDim.x / 2 * blockDim.y / 2);
	if ((i < di) && (j < dj))
	output[(i+di) + (j+dj)*blockDim.x] = input[i + j*blockDim.x];
	if ((i >= di) && (j < dj))
	output[(i - di) + (j + dj)*blockDim.x] = input[i + j*blockDim.x];
	if ((i >= di) && (j >= dj))
	output[(i - di) + (j - dj)*blockDim.x] = input[i + j*blockDim.x];
	if ((i < di) && (j >= dj))
	output[(i + di) + (j - dj)*blockDim.x] = input[i + j*blockDim.x];

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

__global__ void calculate(cuComplex *fths, int *xo, int *yo, double *uo, float *zo2, float dfxs, float lambda, float k0, int Ts, float *fxs, float * y0seg, float* x0seg, int S_Bx, int S_By, int N_Bx, int N_By, int q)
{



	float yp = yo[threadIdx.x] - y0seg[blockIdx.y];

	float xp = xo[threadIdx.x] - x0seg[blockIdx.x];

    float rp = sqrt(zo2[threadIdx.x] + xp*xp + yp*yp);


    float inv_rp = 1 / rp;

    float fxp = xp*inv_rp / lambda;
	float fyp = yp*inv_rp / lambda;



	int iifx = round(fxp / dfxs) + S_Bx / 2 + 1;
	int iify = round(fyp / dfxs) + S_By / 2 + 1;

	if (iifx <= 0 || iifx > S_Bx || iify <= 0 || iify > S_Bx){
		iifx = S_Bx / 2 + 1;
		iify = S_Bx / 2 + 1;
	}



	cuComplex c0;
	cuComplex arg;
	cuComplex arg1;
//	arg.x = (k0*rp - 2 * CUDART_PI_F*(fxs[iifx] + fxs[iify])*(Ts / 2));

//	arg1.x = (2 * CUDART_PI_F  * uo[threadIdx.x] / 6400); 

//	arg1.x = 2;
//	arg.x = -35.699;
	cuComplex res;
	float t = arg1.x*inv_rp;
	sincosf(-arg.x, &res.y, &res.x);
	res.x *= t;
	res.y *= t;

//	c0 = expf(arg);
//	cuComplex uoo = expf(arg1);
//	c0.x = uo[threadIdx.x] * c0.x;
//	c0.y = uo[threadIdx.x] * c0.y;
	c0 = res;


	


	//fths[iifx + blockIdx.x*S_Bx + iify*S_Bx*N_Bx + blockIdx.y* S_Bx*N_Bx*S_By].x += c0.x;
	//fths[iifx + blockIdx.x*S_Bx + iify*S_Bx*N_Bx + blockIdx.y* S_Bx*N_Bx*S_By].y += c0.y;

	fths[iifx + iify*S_Bx + blockIdx.y*S_Bx*S_By + blockIdx.x* S_Bx*N_Bx*S_By].x += c0.x;
	fths[iifx + iify*S_Bx + blockIdx.y*S_Bx*S_By + blockIdx.x* S_Bx*N_Bx*S_By].y += c0.y;

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




void FPAS_CGH_2D(int Np, int* xo, int* yo, int* zo, double* uo, int Nx, int Ny, int dx, float lambda, int S_Bx, int S_By, int q, cuComplex* fths_p, cuComplex* fths_s)
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


	float *z02;
	z02 = (float*)malloc((Np)* sizeof(float));

	for (int t = 0; t < Np; t++)
		z02[t] = zo[t] * zo[t];


	int *d_xo;
	int *d_yo;
	float *d_z02;
	double *d_uo;

	cudaMalloc((void**)&d_xo, sizeof(int)*Np);
	cudaMalloc((void**)&d_yo, sizeof(int)*Np);
	cudaMalloc((void**)&d_z02, sizeof(float)*Np);
	cudaMalloc((void**)&d_uo, sizeof(double)*Np);

	cudaMemcpy(d_xo, xo, Np*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_yo, yo, Np*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_z02, z02, Np*sizeof(float), cudaMemcpyHostToDevice);
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


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time for the kernel: %f ms\n", time);


	dim3 grids;
	grids.x = N_Bx;
	grids.y = N_By;

	dim3 blocks;
	blocks.x = S_Bx;
	blocks.y = S_By;

	cudaEventRecord(start, 0);
	shift2Dout << < grids, blocks >> > (fths_p, fths_s);
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
	int Nx = 1024;
	int Ny = 1024;
	int dx = 8;
	float lambda = 0.5;
	/*START CUDA FFT 2D PART - DEKLARACJE*/
	int S_Bx = 16;
	int S_By = 16;

	cufftComplex* h_out; //dane wynikowe CPU
	cufftComplex* holo; //dane wyjœciowe GPU

	int batch = Nx / S_Bx * Ny / S_By;  //N_Bx*N_By
	cufftHandle forwardPlan;

	preparePlan2D(&forwardPlan, S_Bx, S_By, batch);


	h_out = (cufftComplex*)malloc(sizeof(cufftComplex)*S_Bx*S_By*batch); //allokacja pamiêci na wynik (CPU)

	cudaMalloc(&holo, sizeof(cufftComplex) *S_Bx*S_By*batch); //allokacja pamiêci na dane wyjœciowe (GPU)
	cudaMemset(holo, 0, sizeof(cufftComplex)*S_Bx*S_By*batch); //Wype³nianie zaalokowanej pamiêci zerami (GPU)

	/*END CUDA FFT 2D PART - DEKLARACJE*/

	/*Kod kernela*/
	int Np = 1024;

	int *xo;
	int *yo;
	int *zo;
	double *uo;


	xo = (int*)malloc((Np)* sizeof(int));
	yo = (int*)malloc((Np)* sizeof(int));
	zo = (int*)malloc((Np)* sizeof(int));
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
	zo[foo] = 500e3;
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

	cudaMemcpy(h_out, fths_p, sizeof(cufftComplex)*S_Bx*S_By*batch, cudaMemcpyDeviceToHost);

	/*START CUDA FFT PART */
	execute2D(&forwardPlan, fths_p, holo, CUFFT_FORWARD);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	/*Wyswietlanie modulu/fazy*/
//	cudaMemcpy(h_out, holo, sizeof(cufftComplex)*S_Bx*S_By*batch, cudaMemcpyDeviceToHost);

	
	/*END CUDA FFT PART */

	// Retrieve result from device and store it in host array
	cudaEventElapsedTime(&time, start, stop);
	printf("Time for the kernel: %f ms\n", time);
	printf("Time for the kernel: %f ms\n", h_out[213100].x);

	printf("END \n");


		DataBlock   data;
		CPUBitmap bitmap(Nx, Ny, &data);
		unsigned char    *dev_bitmap;

		HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
		data.dev_bitmap = dev_bitmap;
		cudaMemset(dev_bitmap, 255, bitmap.image_size());

		dim3    grid(Nx, Ny);
		copy2bitmap << <grid, 1 >> >(fths_p, dev_bitmap);
	
		HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
			bitmap.image_size(),
			cudaMemcpyDeviceToHost));

		bitmap.display_and_exit();
	
	return 0;
}

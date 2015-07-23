/*
 * main.cpp
 *
 *  Created on: Jul 4, 2015
 *      Author: markus
 */

#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <numeric>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include <algorithm>
using namespace std::chrono;
using namespace std;

struct Mat {

	Mat() : data(NULL), w(0), h(0) {}
	~Mat() { if (data) delete[] data; }

	float *data;
	int w;
	int h;
};

inline static float getValue(Mat *mat, int x, int y)
{
	if (x > mat->w || y > mat->h) {
		throw runtime_error("invalid access");
	}
	return mat->data[y * mat->w + x];
}

inline static void setValue(Mat *mat, int x, int y, float val)
{
	if (x > mat->w || y > mat->h) {
		throw runtime_error("invalid access");
	}
	mat->data[y * mat->w + x] = val;
}

static void initMat(Mat *mat, int height, int width)
{
	//std::cout << "make matrix (w/h): " << width << "/" << height << std::endl;
	mat->data = new float[height * width];
	mat->w = width;
	mat->h = height;
	for (int i = 0; i < mat->w; i++) {
		for (int j = 0; j < mat->h; j++) {
			setValue(mat, i, j, 0.0f);
		}
	}
}

static void printMat(Mat &mat, bool force = false)
{
	std::cout << "Dim: " << mat.h << ", " << mat.w << "\n";
	if ((mat.w < 10 && mat.h < 10) || force)
	{
		for (int j = 0; j < mat.h; j++) {
			for (int i = 0; i < mat.w; i++) {
				std::cout << getValue(&mat, i, j) << "\t";
			}
			std::cout << "\n";
		}
	}
	std::cout << std::endl;
}

static bool read_csv(string file, Mat *xs, Mat *ys)
{
	ifstream s(file);
	if (!s.is_open()) {
		throw runtime_error(file + " doesn't exist");
	}

	int rows = 0;
	int cols = 0;
	string line;
	while (getline(s, line)) {
		// if we read first line, check how many columns
		if (rows++ == 0) {
			stringstream ss(line);

			while (ss.good()) {
				string substr;
				getline(ss, substr, ',');
				cols++;
			}
		}
	}
	std::cout << "found " << rows << " rows with " << cols << " columns." << std::endl;
	s.clear() ;
	s.seekg(0, ios::beg);

	initMat(xs, rows - 1, cols - 2);
	initMat(ys, rows - 1, 1);


	// go to second line
	getline(s, line);
	int y = 0;
	while (getline(s, line)) {
		stringstream ss(line);

		int x = 0;
		while (ss.good()) {
			string substr;
			getline(ss, substr, ',');

			// first column is uninteresting
			// second column is target values
			if (x == 1) {
				float val = atof(substr.c_str());
				setValue(ys, 0, y, val);
			} else if (x > 1) {
				float val = atof(substr.c_str());
				setValue(xs, (x - 2), y, val);
			}
			x++;
		}
		y++;
	}

	return true;
}

static void min_max_normalize(Mat *m)
{
	for (int x = 0; x < m->w; ++x) {
		// calculate std for each column
		float min = getValue(m, x, 0);
		float max = getValue(m, x, 0);
		for (int y = 1; y < m->h; ++y) {
			float val = getValue(m, x, y);
			if (val < min) {
				min = val;
			} else if (val > max) {
				max = val;
			}
		}

		for (int y = 0; y < m->h; ++y) {
			float val = getValue(m, x, y);
			setValue(m, x, y, (val - min) / max);
		}
	}
}

static void fillRandom(Mat *mat, float LO, float HI)
{
	for (int i = 0; i < mat->w; ++i) {
		for (int j = 0; j < mat->h; ++j) {
			float r = LO + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(HI-LO)));
			setValue(mat, i, j, r);
		}
	}
}

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
#define SAFE_CALL(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

__global__ void matrixMulKernel(float *m1, float *m2, float *r, int m1w, int m2w, int rw, int rh)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < rh) && (col < rw)) {
		// dot product
		float accum = 0.0f;
		for (int c = 0; c < m1w; c++)
		{
			float v1 = m1[row * m1w + c];
			float v2 = m2[c * m2w + col];
			accum += (v1 *  v2);
		}

		r[row * rw + col] = accum;
	}
}

__global__ void sigmoidKernel(float *r, int m)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < m) {
		float val = r[index];
		r[index] = 1.0 / (1.0 + expf(-val));
	}
}

__global__ void matrixAbsErrorKernel(float *p, float *ys, float *r, int rw, int rh)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < rh) && (col < rw)) {
		float pval = p[row * rw + col];
		float ysval = ys[row * rw + col];

		float v = pval - ysval;
		r[row * rw + col] = v * v;
	}
}

__global__ void absErrorKernel(float *p, float *ys, float *r, int m)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < m) {
		float pval = p[index];
		float ysval = ys[index];

		float v = pval - ysval;
		r[index] = v * v;
	}
}

__global__ void updateParamsAbsErrorKernel(float *p, float *ys, float *th, float *xs, int m, float alpha)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < m) {
		float h = *p;
		float y = *ys;

		float x = xs[index];

		th[index] = th[index] - alpha * (h - y) * x;
	}
}

__global__ void crossEntropyKernel(float *p, float *ys, float *r, int m)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < m) {
		float pval = p[index];
		float ysval = ys[index];

		float ex = log1pf(expf(-ysval * pval));
		r[index] = ex;
	}
}

#define REDUCE_BLOCK_SIZE 128
__global__ void reduceKernel(float * input, float * output, int len) {
    //@@ Load a segment of the input vector into shared memory
    __shared__ float partialSum[2 * REDUCE_BLOCK_SIZE];
    unsigned int t = threadIdx.x, start = 2 * blockIdx.x * REDUCE_BLOCK_SIZE;
    if (start + t < len)
       partialSum[t] = input[start + t];
    else
       partialSum[t] = 0;
    if (start + REDUCE_BLOCK_SIZE + t < len)
       partialSum[REDUCE_BLOCK_SIZE + t] = input[start + REDUCE_BLOCK_SIZE + t];
    else
       partialSum[REDUCE_BLOCK_SIZE + t] = 0;
    //@@ Traverse the reduction tree
    for (unsigned int stride = REDUCE_BLOCK_SIZE; stride >= 1; stride >>= 1) {
       __syncthreads();
       if (t < stride)
          partialSum[t] += partialSum[t+stride];
    }
    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index
    if (t == 0)
       output[blockIdx.x] = partialSum[0];
}

static void train_LogRegGpu2(Mat *xs, Mat *ys, Mat *params, Mat *trainedParams, int maxIterations, float alpha, vector<float> &costs)
{
	// put stuff into gpu
	float *gpu_xs;
	float *gpu_ys;

	float *gpu_prediction;

	float *gpu_params;
	float *gpu_abs_error;
	float *gpu_err_cost;

	float *gpu_predictions;
	Mat predictions;
	initMat(&predictions, ys->h, ys->w);

	Mat absErrors;
	initMat(&absErrors, ys->h, ys->w);

	int m = ys->h;

	int numOutputElements;
	numOutputElements = m / (REDUCE_BLOCK_SIZE<<1);
	if (m % (REDUCE_BLOCK_SIZE<<1)) {
		numOutputElements++;
	}

	SAFE_CALL(cudaMalloc((void**)&gpu_xs, sizeof(float) * xs->w * xs->h));
	SAFE_CALL(cudaMalloc((void**)&gpu_ys, sizeof(float) * ys->w * ys->h));
	SAFE_CALL(cudaMalloc((void**)&gpu_prediction, sizeof(float)));
	SAFE_CALL(cudaMalloc((void**)&gpu_predictions, sizeof(float) * ys->w * ys->h));
	SAFE_CALL(cudaMalloc((void**)&gpu_abs_error, sizeof(float) * ys->w * ys->h));
	SAFE_CALL(cudaMalloc((void**)&gpu_params, sizeof(float) * params->w * params->h));
	SAFE_CALL(cudaMalloc((void**)&gpu_err_cost, sizeof(float) * numOutputElements));

	SAFE_CALL(cudaMemcpy(gpu_xs, xs->data, sizeof(float) * xs->w * xs->h, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(gpu_ys, ys->data, sizeof(float) * ys->w * ys->h, cudaMemcpyHostToDevice));
	SAFE_CALL(cudaMemcpy(gpu_params, params->data, sizeof(float) * params->w * params->h, cudaMemcpyHostToDevice));

	// invoke kernel
	static const int blockWidth = 16;
	static const int blockHeight = blockWidth;
	int numBlocksW = xs->w / blockWidth;
	int numBlocksH = xs->h / blockHeight;
	if (xs->w % blockWidth) numBlocksW++;
	if (xs->h % blockHeight) numBlocksH++;

	dim3 dimGrid(numBlocksW, numBlocksH);
	dim3 dimBlock(blockWidth, blockHeight);

	dim3 dimReduce((m - 1) / REDUCE_BLOCK_SIZE + 1);
	dim3 dimReduceBlock(REDUCE_BLOCK_SIZE);

	dim3 dimVectorGrid(((m - 1) / blockWidth * blockWidth) + 1);
	dim3 dimVectorBlock(blockWidth * blockWidth);

	float* error_accum = new float[numOutputElements];
	for (int iter = 0; iter < maxIterations; ++iter) {
		for (int i = 0; i < m; ++i) {
			matrixMulKernel<<<dimGrid, dimBlock>>>(&gpu_xs[i * xs->w], gpu_params, gpu_prediction, xs->w, params->w, 1, 1);
			sigmoidKernel<<<dimVectorGrid, dimVectorBlock>>>(gpu_prediction, 1);
			updateParamsAbsErrorKernel<<<dimVectorGrid, dimVectorBlock>>>(gpu_prediction, &gpu_ys[i], gpu_params, &gpu_xs[i * xs->w], params->h, alpha);
		}
		matrixMulKernel<<<dimGrid, dimBlock>>>(gpu_xs, gpu_params, gpu_predictions, xs->w, params->w, predictions.w, predictions.h);
		sigmoidKernel<<<dimVectorGrid, dimVectorBlock>>>(gpu_predictions, m);


		// calculate error
		absErrorKernel<<<dimVectorGrid, dimVectorBlock>>>(gpu_predictions, gpu_ys, gpu_abs_error, m);
		reduceKernel<<<dimReduce, dimReduceBlock>>>(gpu_abs_error, gpu_err_cost, m);
		SAFE_CALL(cudaMemcpy(error_accum, gpu_err_cost, sizeof(float) * numOutputElements, cudaMemcpyDeviceToHost));
		float g_sum = 0;
		for (int i = 0; i < numOutputElements; ++i)
		{
			g_sum += error_accum[i];
		}

		g_sum /= (2*m);

		costs.push_back(g_sum);

		cout << g_sum << "\n";
	}
	cout << endl;

	delete[] error_accum;
	SAFE_CALL(cudaFree(gpu_xs));
	SAFE_CALL(cudaFree(gpu_ys));
	SAFE_CALL(cudaFree(gpu_abs_error));
	SAFE_CALL(cudaFree(gpu_prediction));
	SAFE_CALL(cudaFree(gpu_predictions));
	SAFE_CALL(cudaFree(gpu_params));
	SAFE_CALL(cudaFree(gpu_err_cost));
}

int main(int argc, char **argv)
{
	string csv_file("./houses.csv");

	Mat xs;
	Mat ys;
	Mat params;
	Mat trainedParams;

	read_csv(csv_file, &xs, &ys);

	//printMat(xs, true);
	//printMat(ys);

	// width of features + 1 for bias
	initMat(&params, xs.w, 1);
	initMat(&trainedParams, xs.w, 1);

	// fill parameter with random initializations from -1 to 1
	fillRandom(&params, -1.0, 1.0);
	//printMat(xs, true);
	//printMat(params, true);

	min_max_normalize(&xs);

	vector<float> costs;

	train_LogRegGpu2(&xs, &ys, &params, &trainedParams, 150, 0.03, costs);

	//for (int i = 0; i < costs.size(); ++i) {
	//	cout << costs[i] << "\n";
	//}

	std::cout << "done" << std::endl;

	//printMat(trainedParams, true);

	return 0;
}


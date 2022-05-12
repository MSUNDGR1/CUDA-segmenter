#include "segmenter.cuh"
#define BLOCK_SIZE 16
#include <stdlib.h>
/*
Images will be stored in row-major (row*rowWidth + column = index) form
*/


__global__ void meanIMG(uint8_t* img, int rows, int cols, float* rowMeans) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < rows) {
		float sum = 0.0;
		for (int i = 0; i < cols; i++) {
			sum += float(img[index * cols + i]);
			if (index == 0) {
				//printf("sum:  %f, i: %d\n", sum, i);
			}
		}
		rowMeans[index] = sum / cols;
		if (index == 0) {
			//printf("rowMean: %f", rowMeans[index]);
		}
	}
}

__global__ void varIMG(uint8_t* img, int rows, int cols, float mean, float* varsums) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < rows) {
		float sum = 0.0;
		for (int i = 0; i < cols; i++) {
			sum += pow(float(img[index * cols + i]) - mean, 2);
		}
		varsums[index] = sum / cols;
	}
}


__global__ void laplace(unsigned char* img, unsigned char* imgOut, int rows, int cols) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols) {
		int maxInd = rows * cols - 1;
		int sum = 0;
		sum += 4 * img[realInd];
		if (rowInd > 0) {
			realInd = (rowInd - 1) * cols + colInd;

			sum += -1 * img[realInd];
		}
		if (rowInd < rows - 1) {
			realInd = (rowInd + 1) * cols + colInd;
			sum += -1 * img[realInd];
		}
		if (colInd > 0) {
			realInd = rowInd * cols + (colInd - 1);

			sum += -1 * img[realInd];
		}
		if (colInd < cols - 1) {
			realInd = rowInd * cols + (colInd + 1);

			sum += -1 * img[realInd];
		}
		realInd = rowInd * cols + colInd;

		if (sum < 255) {
			imgOut[realInd] = (unsigned char)sum;
		}
		else {
			imgOut[realInd] = 255;
		}
		//printf("realInd: %d, col: %d\n", realInd, colInd);
	}
}



__global__ void ngbrCompLEA(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols && colInd != 0 && colInd % 2 == 0) {
		int nextInd = rowInd * cols + (colInd - 1);
		float currClustMean = clusterMeans[clusters[realInd]];
		if ((clusters[nextInd] = nextInd && abs(currClustMean - int(img[nextInd])) < diff)
			|| abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))){
		
			newClust[nextInd] = clusters[realInd];
			
		}
	}
}

__global__ void ngbrCompLOA(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols && colInd != 0 && colInd % 2 == 1) {
		int nextInd = rowInd * cols + (colInd - 1);
		float currClustMean = clusterMeans[clusters[realInd]];
		if ((clusters[nextInd] = nextInd && abs(currClustMean - int(img[nextInd])) < diff)
			|| abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {

			newClust[nextInd] = clusters[realInd];

		}
	}
}

__global__ void ngbrCompREA(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols - 1 && colInd % 2 == 0) {
		int nextInd = rowInd * cols + (colInd + 1);
		float currClustMean = clusterMeans[clusters[realInd]];
		if ((clusters[nextInd] = nextInd && abs(currClustMean - int(img[nextInd])) < diff)
			|| abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {

			newClust[nextInd] = clusters[realInd];

		}
	}
}

__global__ void ngbrCompROA(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols - 1 && colInd % 2 == 1) {
		int nextInd = rowInd * cols + (colInd + 1);
		float currClustMean = clusterMeans[clusters[realInd]];
		if ((clusters[nextInd] = nextInd && abs(currClustMean - int(img[nextInd])) < diff)
			|| abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {

			newClust[nextInd] = clusters[realInd];

		}
	}
}

__global__ void ngbrCompUEA(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols && rowInd != 0 && rowInd % 2 == 0) {
		int nextInd = (rowInd - 1) * cols + (colInd);
		float currClustMean = clusterMeans[clusters[realInd]];
		if ((clusters[nextInd] = nextInd && abs(currClustMean - int(img[nextInd])) < diff)
			|| abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {

			newClust[nextInd] = clusters[realInd];

		}
	}
}

__global__ void ngbrCompUOA(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols && rowInd != 0 && rowInd % 2 == 1) {
		int nextInd = (rowInd - 1) * cols + (colInd);
		float currClustMean = clusterMeans[clusters[realInd]];
		if ((clusters[nextInd] = nextInd && abs(currClustMean - int(img[nextInd])) < diff)
			|| abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {

			newClust[nextInd] = clusters[realInd];

		}
	}
}

__global__ void ngbrCompDEA(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows-1 && colInd < cols && rowInd != 0 && rowInd % 2 == 0) {
		int nextInd = (rowInd + 1) * cols + (colInd);
		float currClustMean = clusterMeans[clusters[realInd]];
		if ((clusters[nextInd] = nextInd && abs(currClustMean - int(img[nextInd])) < diff)
			|| abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {

			newClust[nextInd] = clusters[realInd];

		}
	}
}

__global__ void ngbrCompDOA(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows - 1 && colInd < cols && rowInd != 0 && rowInd % 2 == 1) {
		int nextInd = (rowInd + 1) * cols + (colInd);
		float currClustMean = clusterMeans[clusters[realInd]];
		if ((clusters[nextInd] = nextInd && abs(currClustMean - int(img[nextInd])) < diff)
			|| abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {

			newClust[nextInd] = clusters[realInd];

		}
	}
}


__global__ void ngbrCompLUE(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols && colInd != 0 && colInd % 2 == 0 && rowInd != 0 && rowInd %2==0) {
		int nextInd = (rowInd-1) * cols + (colInd - 1);
		float currClustMean = clusterMeans[clusters[realInd]];
		if (abs(currClustMean - int(img[nextInd])) < diff && clusters[realInd] != clusters[nextInd]) {
			if (clusters[nextInd] = nextInd || abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {
				newClust[nextInd] = clusters[realInd];
			}
		}
	}
}

__global__ void ngbrCompRUE(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols-1  && colInd % 2 == 0 && rowInd != 0 && rowInd % 2 == 0) {
		int nextInd = (rowInd -1) * cols + (colInd + 1);
		float currClustMean = clusterMeans[clusters[realInd]];
		if (abs(currClustMean - int(img[nextInd])) < diff && clusters[realInd] != clusters[nextInd]) {
			if (clusters[nextInd] = nextInd || abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {
				newClust[nextInd] = clusters[realInd];
			}
		}
	}
}

__global__ void ngbrCompLDE(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows-1 && colInd < cols && colInd != 0 && colInd % 2 == 0 && rowInd % 2 == 0) {
		int nextInd = (rowInd + 1) * cols + (colInd - 1);
		float currClustMean = clusterMeans[clusters[realInd]];
		if (abs(currClustMean - int(img[nextInd])) < diff && clusters[realInd] != clusters[nextInd]) {
			if (clusters[nextInd] = nextInd || abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {
				newClust[nextInd] = clusters[realInd];
			}
		}
	}
}

__global__ void ngbrCompRDE(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows-1 && colInd < cols-1 && colInd % 2 == 0 && rowInd % 2 == 0) {
		int nextInd = (rowInd + 1) * cols + (colInd + 1);
		float currClustMean = clusterMeans[clusters[realInd]];
		if (abs(currClustMean - int(img[nextInd])) < diff && clusters[realInd] != clusters[nextInd]) {
			if (clusters[nextInd] = nextInd || abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {
				newClust[nextInd] = clusters[realInd];
			}
		}
	}
}


__global__ void ngbrCompLE(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols && colInd !=0 && colInd %2 ==0) {
		int nextInd = rowInd * cols + (colInd - 1);
		float currClustMean = clusterMeans[clusters[realInd]];
		if (abs(currClustMean - int(img[nextInd])) < diff && clusters[realInd] != clusters[nextInd]) {
			if (clusters[nextInd] = nextInd || abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {
				newClust[nextInd] = clusters[realInd];
			}
		}
	}
}

__global__ void ngbrCompLO(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols && colInd != 0 && colInd % 2 == 1) {
		int nextInd = rowInd * cols + (colInd - 1);
		float currClustMean = clusterMeans[clusters[realInd]];
		if (abs(currClustMean - int(img[nextInd])) < diff && clusters[realInd] != clusters[nextInd]) {
			if (clusters[nextInd] = nextInd || abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {
				newClust[nextInd] = clusters[realInd];
			}
		}
	}
}

__global__ void ngbrCompRE(unsigned char* img, int* clusters, int* clusterSize,  int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;



	if (rowInd < rows && colInd < cols -1 && colInd % 2 == 0) {
		int nextInd = rowInd * cols + (colInd + 1);
		
		float currClustMean = clusterMeans[clusters[realInd]];
		
		if (abs(currClustMean - int(img[nextInd])) < diff && clusters[realInd] != clusters[nextInd]) {
			if (clusters[nextInd] = nextInd || abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {
				newClust[nextInd] = clusters[realInd];
			}
		}
	}
}

__global__ void ngbrCompRO(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;



	if (rowInd < rows && colInd < cols - 1 && colInd % 2 == 1) {
		int nextInd = rowInd * cols + (colInd + 1);

		float currClustMean = clusterMeans[clusters[realInd]];

		if (abs(currClustMean - int(img[nextInd])) < diff && clusters[realInd] != clusters[nextInd]) {
			if (clusters[nextInd] = nextInd || abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {
				newClust[nextInd] = clusters[realInd];
			}
		}
	}
}

__global__ void ngbrCompUE(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;



	if (rowInd < rows && colInd < cols && rowInd != 0 && rowInd % 2 == 0) {
		int nextInd = (rowInd-1) * cols + (colInd);
		
		float currClustMean = clusterMeans[clusters[realInd]];
		if (abs(currClustMean - img[nextInd]) < diff && clusters[realInd] != clusters[nextInd]) {
			if (clusters[nextInd] = nextInd || abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {
				newClust[nextInd] = clusters[realInd];
				if (rowInd > 180 && rowInd < 200 && colInd > 180 && colInd < 200) {
					//printf("grouped");
				}
			}
			
		}
	}
}

__global__ void ngbrCompDE(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;



	if (rowInd < rows-1 && colInd < cols && rowInd % 2 == 0) {
		int nextInd = (rowInd + 1) * cols + (colInd);

		float currClustMean = clusterMeans[clusters[realInd]];
		if (abs(currClustMean - img[nextInd]) < diff && clusters[realInd] != clusters[nextInd]) {
			if (clusters[nextInd] = nextInd || abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {
				newClust[nextInd] = clusters[realInd];
			}
		}
	}
}

__global__ void ngbrCompUO(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;



	if (rowInd < rows && colInd < cols && rowInd != 0 && rowInd % 2 == 1) {
		int nextInd = (rowInd - 1) * cols + (colInd);

		float currClustMean = clusterMeans[clusters[realInd]];
		if (abs(currClustMean - img[nextInd]) < diff && clusters[realInd] != clusters[nextInd]) {
			if (clusters[nextInd] = nextInd || abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {
				newClust[nextInd] = clusters[realInd];
			}

		}
	}
}

__global__ void ngbrCompDO(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;



	if (rowInd < rows-1 && colInd < cols && rowInd % 2 == 1) {
		int nextInd = (rowInd + 1) * cols + (colInd);

		float currClustMean = clusterMeans[clusters[realInd]];
		if (abs(currClustMean - img[nextInd]) < diff && clusters[realInd] != clusters[nextInd]) {
			if (clusters[nextInd] = nextInd || abs(currClustMean - int(img[nextInd])) < abs(clusterMeans[clusters[nextInd]] - int(img[nextInd]))) {
				newClust[nextInd] = clusters[realInd];
			}
		}
	}
}

__global__ void ngbrComp(unsigned char* img, int* clusters, int* clusterSize, bool setup, int rows, int cols, float diff, int* newClust, float* clusterMeans) {

	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	

	if (rowInd < rows && colInd < cols) {
		float minDiff = 255.0;
		if (setup) {
			clusters[realInd] = realInd;
			clusterSize[realInd] = 1;

		}
		int nextInd;
		float currClustMean = clusterMeans[clusters[realInd]];
		/*if (rowInd > 0) {
			nextInd = (rowInd - 1) * cols + colInd;

			if (abs(currClustMean - img[nextInd]) < minDiff && clusters[realInd] != clusters[nextInd]) {
				minDiff = abs(img[realInd] - img[nextInd]);
				groupMerge = nextInd;
			}
		}
		if (rowInd < rows - 1) {
			nextInd = (rowInd + 1) * cols + colInd;

			if (abs(currClustMean - img[nextInd]) < minDiff && clusters[realInd] != clusters[nextInd]) {
				minDiff = abs(img[realInd] - img[nextInd]);
				groupMerge = nextInd;
			}
		}
		if (colInd > 0) {
			nextInd = rowInd * cols + (colInd - 1);

			if (abs(currClustMean - img[nextInd]) < minDiff && clusters[realInd] != clusters[nextInd]) {
				minDiff = abs(img[realInd] - img[nextInd]);
				groupMerge = nextInd;
			}
		}
		if (colInd < cols - 1) {
			nextInd = rowInd * cols + (colInd + 1);

			if (abs(currClustMean - img[nextInd]) < minDiff && clusters[realInd] != clusters[nextInd]) {
				minDiff = abs(img[realInd] - img[nextInd]);
				groupMerge = nextInd;
			}
		}
		
		if (abs(currClustMean - img[groupMerge]) < diff) {
			if (clusterSize[groupMerge] > clusterSize[realInd]) {
				newClust[realInd] = clusters[groupMerge];
			}else {
				newClust[groupMerge] = clusters[realInd];
			}
		}*/
		
		if (rowInd < rows - 1) {
			nextInd = (rowInd + 1) * cols + colInd;

			if (abs(currClustMean - img[nextInd]) < diff && clusters[realInd] != clusters[nextInd]) {
				newClust[nextInd] = clusters[realInd];
				minDiff = abs(currClustMean - img[nextInd]);
			}
		}
		if (colInd < cols - 1) {
			nextInd = rowInd * cols + (colInd + 1);

			if (abs(currClustMean - img[nextInd]) < diff && clusters[realInd] != clusters[nextInd]) {
				if (abs(currClustMean - img[nextInd]) < minDiff) {
					newClust[nextInd] = clusters[realInd];
				}
			}
		}
	}

}

__global__ void imOut(unsigned char* imgOut, int* clusters, float* clusterMeans, int rows, int cols) {
	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols) {
		imgOut[realInd] = (unsigned char)clusterMeans[clusters[realInd]];
		
	}
}

__global__ void imPProc(unsigned char* imgOut, int* clusters, float* clusterMeans, int rows, int cols, int* clusterSize, int minSize) {
	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols) {
		if (clusterSize[clusters[realInd]] > minSize) {
			imgOut[realInd] = (unsigned char)clusterMeans[clusters[realInd]];
		}
		else {
			imgOut[realInd] = 0;
		}
	}
}

__global__ void clusterSync(int* clusters, int rows, int cols, int* newClust, int* clusterSize, float* clustMeans) {
	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols) {
		clusters[realInd] = newClust[realInd];
		//clusterSize[realInd] = 0;
		//clustMeans[realInd] = 0.0;
	}
}

__global__ void clusterMeanMul(int* clusterSize, float* clusterMeans, int rows, int cols) {
	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols) {
		clusterMeans[realInd] = clusterMeans[realInd] * clusterSize[realInd];
	}
}

__global__ void clusterMeanSum(unsigned char* img, int* clusters, int* clusterSize, int rows, int cols, float* clusterMeans, int* newClust) {
	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols) {
		if (clusters[realInd] != newClust[realInd]) {
			float* address = clusterMeans + newClust[realInd];
			atomicAdd(address, float(img[realInd]));
			address = clusterMeans + clusters[realInd];
			atomicAdd(address, -1.0 * float(img[realInd]));
			int* sizeAddress = clusterSize + newClust[realInd];
			atomicAdd(sizeAddress, 1);
			sizeAddress = clusterSize + clusters[realInd];
			atomicAdd(sizeAddress, -1);
		}
		//printf("added to cluster: %d,  val: %f\n", clusters[realInd], float(img[realInd]));
		//printf("realInd: %d,  addy: %p, cluster: %d\n",  realInd, address, clusters[realInd]);
	}
}

__global__ void clusterMeanInit(int rows, int cols, float* clusterMeans, int* clusterSize, int* clusters, int* newClusters, unsigned char* img) {
	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols) {
		clusters[realInd] = realInd;
		//int clustid = clusters[realInd];
		newClusters[realInd] = realInd;
		clusterSize[realInd] = 0;
		int* address = clusterSize + (realInd);
		atomicAdd(address, 1);
		clusterMeans[realInd] = img[realInd];
		//printf("cluster size: %d, realInd: %d, address: %p, clusterSIze addy: %p\n", clusterSize[realInd], realInd, address, clusterSize);
		//if (rowInd > 180 && rowInd < 200 && colInd < 200 && colInd > 180) {
			//printf("img: %d", img[realInd]);
		//}
	}
}

__global__ void clusterMeanDiv(float* clusterMeans, int rows, int cols, int* clusterSize) {
	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols) {
		if (clusterSize[realInd] != 0) {
			clusterMeans[realInd] = clusterMeans[realInd] / clusterSize[realInd];//check if cluster size not zero
		}
		else {
			clusterMeans[realInd] = 0.0;
		}
		//printf("clusterMean: %f,  realInd: %d,  clusterSize: %d\n", clusterMeans[realInd], realInd, clusterSize[realInd]);
		//if (clusterSize[realInd] > 10) {
			//printf("largeCluster\n");
		//}
	}
}

__global__ void clusterIsolateKernel(unsigned char* Rchan, unsigned char* Gchan, unsigned char* Bchan, int* clusterIDS, int rows, int cols, char R, char G, char B, int clusterNum) {
	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols) {
		if (clusterIDS[realInd] == clusterNum) {
			Rchan[realInd] = R;
			Gchan[realInd] = G;
			Bchan[realInd] = B;
		}
	}
}


__global__ void clusterBoundaryDraw(unsigned char* d_img, int* clusters, int rows, int cols, float* clustMeans, float diff, bool allSeg) {
	int rowInd = blockIdx.x * blockDim.x + threadIdx.x;
	int colInd = blockIdx.y * blockDim.y + threadIdx.y;
	int realInd = rowInd * cols + colInd;

	if (rowInd < rows && colInd < cols) {
		bool border = false;
		if (allSeg) {
			if (rowInd < rows - 1) {
				int nextInd = (rowInd + 1) * cols + (colInd);
				if (clusters[nextInd] != clusters[realInd]) {
					border = true;
				}
			}
			if (rowInd != 0) {
				int nextInd = (rowInd - 1) * cols + (colInd);
				if (clusters[nextInd] != clusters[realInd]) {
					border = true;
				}
			}
			if (colInd < cols - 1) {
				int nextInd = (rowInd) * cols + (colInd+1);
				if (clusters[nextInd] != clusters[realInd]) {
					border = true;
				}
			}
			if (colInd != 0) {
				int nextInd = (rowInd) * cols + (colInd-1);
				if (clusters[nextInd] != clusters[realInd]) {
					border = true;
				}
			}
		}
		else {
			if (rowInd < rows - 1) {
				int nextInd = (rowInd + 1) * cols + (colInd);
				if (clusters[nextInd] != clusters[realInd] && abs(clustMeans[clusters[nextInd]] - clustMeans[clusters[realInd]]) > diff) {
					border = true;
				}
			}
			if (rowInd != 0) {
				int nextInd = (rowInd - 1) * cols + (colInd);
				if (clusters[nextInd] != clusters[realInd] && abs(clustMeans[clusters[nextInd]] - clustMeans[clusters[realInd]]) > diff) {
					border = true;
				}
			}
			if (colInd < cols - 1) {
				int nextInd = (rowInd)*cols + (colInd + 1);
				if (clusters[nextInd] != clusters[realInd] && abs(clustMeans[clusters[nextInd]] - clustMeans[clusters[realInd]]) > diff) {
					border = true;
				}
			}
			if (colInd != 0) {
				int nextInd = (rowInd)*cols + (colInd - 1);
				if (clusters[nextInd] != clusters[realInd] && abs(clustMeans[clusters[nextInd]] - clustMeans[clusters[realInd]]) > diff) {
					border = true;
				}
			}
		}
		if (border) {
			d_img[realInd] = 0;
		}
		else {
			d_img[realInd] = 255;
		}
	}
}
using namespace cv;

namespace seg {

	Mat clusterIsolate(Mat input, int* clusterIDs, char R, char G, char B, int ClusterNum) {
		std::vector<uchar> RmVec((input.rows * input.cols), 0);
		std::vector<uchar> GmVec((input.rows * input.cols), 0);
		std::vector<uchar> BmVec((input.rows * input.cols), 0);
		//if (inputBW.isContinuous()) {
			//mVec.assign(inputBW.data, inputBW.data + inputBW.total());
		//}
		//else {
		for (int r = 0; r < input.rows; r++) {
			for (int c = 0; c < input.cols; c++) {
				int index = input.cols * r + c;
				RmVec[index] = input.at<uchar>(r, c);
				GmVec[index] = input.at<uchar>(r, c);
				BmVec[index] = input.at<uchar>(r, c);
			}
		}
		unsigned char* d_imgR, * d_imgG,* d_imgB;
		
		int * d_clusterIDs;
		int size = sizeof(unsigned char) * input.rows * input.cols;
		cudaMalloc((void**)&d_imgR, size); cudaMalloc((void**)&d_imgG, size); cudaMalloc((void**)&d_imgB, size);
		cudaMemcpy(d_imgR, (unsigned char*)&RmVec[0], size, cudaMemcpyHostToDevice); cudaMemcpy(d_imgG, (unsigned char*)&GmVec[0], size, cudaMemcpyHostToDevice); cudaMemcpy(d_imgB, (unsigned char*)&BmVec[0], size, cudaMemcpyHostToDevice);
		size = sizeof(int) * input.rows * input.cols;
		cudaMalloc((void**)&d_clusterIDs, size); 
		cudaMemcpy(d_clusterIDs, clusterIDs, size, cudaMemcpyHostToDevice);
		
		
		//printf("variance: %f,  diff: %f\n", imStat.first, diff);
		int numBlocksH = input.cols / BLOCK_SIZE;
		if (input.cols % BLOCK_SIZE > 0) {
			numBlocksH++;
		}
		int numBlocksV = input.rows / BLOCK_SIZE;
		if (input.rows % BLOCK_SIZE > 0) {
			numBlocksV++;
		}
		dim3 gGrid(numBlocksV, numBlocksH);
		dim3 bGrid(BLOCK_SIZE, BLOCK_SIZE);
		printf("Kernel Call \n");
		clusterIsolateKernel << <gGrid, bGrid >> > (d_imgR, d_imgG, d_imgB, d_clusterIDs, input.rows, input.cols, R, G, B, ClusterNum);
		printf("Isolated \n");
		cudaFree(d_clusterIDs);
		size = sizeof(char) * input.rows * input.cols;
		cudaMemcpy((unsigned char*)&RmVec[0], d_imgR,  size, cudaMemcpyDeviceToHost); cudaMemcpy((unsigned char*)&GmVec[0], d_imgG,  size, cudaMemcpyDeviceToHost); cudaMemcpy((unsigned char*)&BmVec[0], d_imgB,  size, cudaMemcpyDeviceToHost);
		cudaFree(d_imgR); cudaFree(d_imgG); cudaFree(d_imgB);
		Mat retImg = Mat(input.rows, input.cols, CV_8UC3);
		for (int r = 0; r < input.rows; r++) {
			for (int c = 0; c < input.cols; c++) {
				int index = input.cols * r + c;
				retImg.at<Vec3b>(r, c).val[0] = RmVec[index];
				retImg.at<Vec3b>(r, c).val[1] = GmVec[index];
				retImg.at<Vec3b>(r, c).val[2] = BmVec[index];
			}
		}
		printf("Returning \n");
		return retImg;
	}

	Mat laplaceEdge(cv::Mat inputBW) {

		std::vector<uchar> mVec((inputBW.rows * inputBW.cols), 0);
		//if (inputBW.isContinuous()) {
			//mVec.assign(inputBW.data, inputBW.data + inputBW.total());
		//}
		//else {
		for (int r = 0; r < inputBW.rows; r++) {
			for (int c = 0; c < inputBW.cols; c++) {
				int index = inputBW.cols * r + c;
				mVec[index] = inputBW.at<uchar>(r, c);
			}
		}
		//}

		unsigned char* d_img, * d_imgN;
		int size = sizeof(unsigned char) * inputBW.rows * inputBW.cols;
		cudaMalloc((void**)&d_img, size); cudaMalloc((void**)&d_imgN, size);
		cudaMemcpy(d_img, (unsigned char*)&mVec[0], size, cudaMemcpyHostToDevice);

		int numBlocksH = inputBW.cols / BLOCK_SIZE;
		if (inputBW.cols % BLOCK_SIZE > 0) {
			numBlocksH++;
		}
		int numBlocksV = inputBW.rows / BLOCK_SIZE;
		if (inputBW.rows % BLOCK_SIZE > 0) {
			numBlocksV++;
		}
		dim3 gGrid(numBlocksV, numBlocksH);
		dim3 bGrid(BLOCK_SIZE, BLOCK_SIZE);

		//printf("row blocks: %d,  col blocks: %d\n", numBlocksV, numBlocksH);
		laplace << <gGrid, bGrid >> > (d_img, d_imgN, inputBW.rows, inputBW.cols);

		cudaMemcpy((unsigned char*)&mVec[0], d_imgN, size, cudaMemcpyDeviceToHost);
		cudaFree(d_img); cudaFree(d_imgN);
		for (int r = 0; r < inputBW.rows; r++) {
			for (int c = 0; c < inputBW.cols; c++) {
				int index = (inputBW.cols * r) + c;

				inputBW.at<uchar>(r, c) = mVec[index];
			}
		}
		return inputBW;
	}


	float mean(cv::Mat input) {

		uint8_t* imgptr;
		float* rowMeans;
		//printf("gpumat start\n");
		//cuda::GpuMat gpuInput(sizeof(unsigned char)*input.rows*input.cols, CV_8UC1);
		cuda::GpuMat gpuInput(input.rows, input.cols, CV_8UC1);
		//printf("gpumat created\n");

		gpuInput.upload(input);
		cudaMalloc((void**)&imgptr, sizeof(unsigned char) * input.rows * input.cols);
		cudaMalloc((void**)&rowMeans, sizeof(float) * input.rows);

		cudaMemcpy(imgptr, gpuInput.ptr<uint8_t>(), sizeof(unsigned char) * input.rows * input.cols, cudaMemcpyDeviceToDevice);
		//printf("gpumat sent to cudamem\n");
		int numBlocks = input.rows / BLOCK_SIZE;
		if (input.rows % BLOCK_SIZE > 0) {
			numBlocks++;
		}
		meanIMG << <numBlocks, BLOCK_SIZE >> > (imgptr, input.rows, input.cols, rowMeans);
		std::vector<float> rowMeansC(input.rows, 0.0);
		cudaMemcpy((float*)&rowMeansC[0], rowMeans, sizeof(float) * input.rows, cudaMemcpyDeviceToHost);
		printf("successful vector copy\n");
		float totMean = 0.0;
		for (int i = 0; i < rowMeansC.size(); i++) {
			totMean += rowMeansC[i];
		}
		totMean /= rowMeansC.size();

		cudaFree(imgptr); cudaFree(rowMeans);
		
		return totMean;
	}
	
	std::pair<float,float> variance(Mat input) {
		uint8_t* imgptr;
		float* rowMeans;
		
		cuda::GpuMat gpuInput(input.rows, input.cols, CV_8UC1);
		

		gpuInput.upload(input);
		cudaMalloc((void**)&imgptr, sizeof(unsigned char) * input.rows * input.cols);
		cudaMalloc((void**)&rowMeans, sizeof(float) * input.rows);

		cudaMemcpy(imgptr, gpuInput.ptr<uint8_t>(), sizeof(unsigned char) * input.rows * input.cols, cudaMemcpyDeviceToDevice);
		//printf("gpumat sent to cudamem\n");
		int numBlocks = input.rows / BLOCK_SIZE;
		if (input.rows % BLOCK_SIZE > 0) {
			numBlocks++;
		}
		meanIMG << <numBlocks, BLOCK_SIZE >> > (imgptr, input.rows, input.cols, rowMeans);
		std::vector<float> rowMeansC(input.rows, 0.0);
		cudaMemcpy((float*)&rowMeansC[0], rowMeans, sizeof(float) * input.rows, cudaMemcpyDeviceToHost);
		printf("successful vector copy\n");
		float totMean = 0.0;
		for (int i = 0; i < rowMeansC.size(); i++) {
			totMean += rowMeansC[i];
		}
		totMean /= rowMeansC.size();

		cudaFree(rowMeans);

		float* varSums; std::vector<float> varSumsC(input.rows, 0.0);
		cudaMalloc((void**)&varSums, sizeof(float) * input.rows);
		varIMG << <numBlocks, BLOCK_SIZE >> > (imgptr, input.rows, input.cols, totMean, varSums);
		cudaMemcpy((float*)&varSumsC[0], varSums, sizeof(float) * input.rows, cudaMemcpyDeviceToHost);

		float varSum = 0.0;
		for (auto val : varSumsC) {
			varSum += val;
		}
		varSum /= varSumsC.size();
		cudaFree(imgptr); cudaFree(varSums);
		return std::make_pair(varSum, totMean);
	}

	std::pair<float, float> varNOMEM(uint8_t* imgptr, int rows, int cols) {
		float* rowMeans;
		int numBlocks =rows / BLOCK_SIZE;
		if (rows % BLOCK_SIZE > 0) {
			numBlocks++;
		}
		cudaMalloc((void**)&rowMeans, sizeof(float) * rows);
		meanIMG << <numBlocks, BLOCK_SIZE >> > (imgptr, rows, cols, rowMeans);
		std::vector<float> rowMeansC(rows, 0.0);
		cudaMemcpy((float*)&rowMeansC[0], rowMeans, sizeof(float) * rows, cudaMemcpyDeviceToHost);
		//printf("successful vector copy\n");
		float totMean = 0.0;
		for (int i = 0; i < rowMeansC.size(); i++) {
			totMean += rowMeansC[i];
		}
		totMean /= rowMeansC.size();

		cudaFree(rowMeans);

		float* varSums; std::vector<float> varSumsC(rows, 0.0);
		cudaMalloc((void**)&varSums, sizeof(float) * rows);
		varIMG << <numBlocks, BLOCK_SIZE >> > (imgptr, rows, cols, totMean, varSums);
		cudaMemcpy((float*)&varSumsC[0], varSums, sizeof(float) * rows, cudaMemcpyDeviceToHost);
		cudaFree(varSums);
		float varSum = 0.0;
		for (auto val : varSumsC) {
			varSum += val;
		}
		varSum /= varSumsC.size();
		return std::make_pair(varSum, totMean);
	}

	std::pair<Mat, int*> segment(Mat input, float diffParam, int cycles) {
		std::vector<uchar> mVec((input.rows * input.cols), 0);
		//if (inputBW.isContinuous()) {
			//mVec.assign(inputBW.data, inputBW.data + inputBW.total());
		//}
		//else {
		for (int r = 0; r < input.rows; r++) {
			for (int c = 0; c < input.cols; c++) {
				int index = input.cols * r + c;
				mVec[index] = input.at<uchar>(r, c);
			}
		}
		//}

		unsigned char* d_img;
		float* clustMean;
		int* clusterSize, * clusterIDs, * newIDs;
		int size = sizeof(unsigned char) * input.rows * input.cols;
		cudaMalloc((void**)&d_img, size);
		cudaMemcpy(d_img, (unsigned char*)&mVec[0], size, cudaMemcpyHostToDevice);
		size = sizeof(int) * input.rows * input.cols;
		cudaMalloc((void**)&clusterSize, size); cudaMalloc((void**)&clusterIDs, size); cudaMalloc((void**)&newIDs, size);
		size = sizeof(float) * input.rows * input.cols;
		cudaMalloc((void**)&clustMean, size);
		std::pair<float, float> imStat = varNOMEM(d_img, input.rows, input.cols);
		float diff = imStat.first * diffParam;
		//printf("variance: %f,  diff: %f\n", imStat.first, diff);
		int numBlocksH = input.cols / BLOCK_SIZE;
		if (input.cols % BLOCK_SIZE > 0) {
			numBlocksH++;
		}
		int numBlocksV = input.rows / BLOCK_SIZE;
		if (input.rows % BLOCK_SIZE > 0) {
			numBlocksV++;
		}
		dim3 gGrid(numBlocksV, numBlocksH);
		dim3 bGrid(BLOCK_SIZE, BLOCK_SIZE);

		clusterMeanInit << <gGrid, bGrid >> > (input.rows, input.cols, clustMean, clusterSize, clusterIDs, newIDs, d_img);
		/*
		bool vec:
		0: LE, 1: RE, 2: LO, 3: RO, 4: UE, 5: DE, 6: UO, 7: DO
		*/
		bool altRandDet = true;
		int altRandDet_detcount = 10;
		int alrRandDet_randcount = 20;
		bool random = false;
		bool postProc = false;
		int minSize = 10;
		bool altMerge = false;
		bool modOrder = false;
		float diffMod = 0.97;
		float oldDiff = diffParam;
		std::vector<bool> segVec(12, false);
		for (int i = 0; i < cycles; i++) {
			if (altRandDet) {
				if (i % (altRandDet_detcount + alrRandDet_randcount) < alrRandDet_randcount) {
					random = true;
				}
				else {
					random = false;
				}
			}
			//diffParam *= diffMod;
			if (diffParam < oldDiff /5.0) {
				//diffParam = oldDiff;
			}
			if (random) {
				for (int j = 0; j < 8; j++) {
					segVec[j] = rand() % 3 == 1;
				}
				if (segVec[0]) {
					ngbrCompLE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					
					clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				}
				if (segVec[1]) {
					ngbrCompRE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					
					clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				}
				if (segVec[4]) {
					ngbrCompUE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					
					clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				}
				if (segVec[5]) {
					ngbrCompDE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					
					clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				}
				if (segVec[2]) {
					ngbrCompLO << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					
					clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				}
				if (segVec[3]) {
					ngbrCompRO << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					
					clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				}
				if (segVec[6]) {
					ngbrCompUO << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					
					clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				}
				if (segVec[7]) {
					ngbrCompDO << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);

				}
				if (segVec[8]) {
					ngbrCompLUE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					
					clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				}
				if (segVec[9]) {
					ngbrCompLDE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					
					clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				}
				if (segVec[10]) {
					ngbrCompRUE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					
					clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				}
				if (segVec[11]) {
					ngbrCompRDE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					
					clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				}
			}
			/*else if (altMerge) {
				ngbrCompLEA << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

				clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
				cudaDeviceSynchronize();
				clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);


				ngbrCompREA << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

				clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);

				clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
				cudaDeviceSynchronize();
				clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);

				ngbrCompUEA << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

				clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);

				clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
				cudaDeviceSynchronize();
				clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);

				ngbrCompDEA << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

				clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);

				clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
				cudaDeviceSynchronize();
				clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);

				ngbrCompLOA << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

				clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);

				clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
				cudaDeviceSynchronize();
				clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);

				ngbrCompROA << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

				clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);

				clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
				cudaDeviceSynchronize();
				clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);

				ngbrCompUOA << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

				clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);

				clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
				cudaDeviceSynchronize();
				clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);

				ngbrCompDOA << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

				clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);

				clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
				cudaDeviceSynchronize();
				clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
			}//
			/*else if (!modOrder) {
					ngbrCompLE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
					cudaDeviceSynchronize();
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
					

					ngbrCompRE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);

					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
					cudaDeviceSynchronize();
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);

					ngbrCompUE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);

					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
					cudaDeviceSynchronize();
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);

					ngbrCompDE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);

					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
					cudaDeviceSynchronize();
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);

					ngbrCompLO << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);

					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
					cudaDeviceSynchronize();
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);

					ngbrCompRO << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);

					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
					cudaDeviceSynchronize();
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);

					ngbrCompUO << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);

					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
					cudaDeviceSynchronize();
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);

					ngbrCompDO << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

					clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);

					clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
					cudaDeviceSynchronize();
					clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);

			}*/
			else {

				ngbrCompRE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

				
				clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
				clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
				clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
				clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				ngbrCompLE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

				
				clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
				clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
				clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
				clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				ngbrCompDE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

				
				clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
				clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
				clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
				clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				ngbrCompUE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

				
				clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
				clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
				clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
				clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				ngbrCompRO << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

				
				clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
				clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
				clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
				clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				ngbrCompLO << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

				
				clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
				clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
				clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
				clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				ngbrCompDO << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

				
				clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
				clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
				clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
				clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
				ngbrCompUO << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

				
				clusterMeanMul << <gGrid, bGrid >> > (clusterSize, clustMean, input.rows, input.cols);
				clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
				clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
				clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
					
			}
		}
			

		//printf("segment iter:%d \n", i);
		//clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
		//clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean);
		//clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
		if (postProc) {
			imPProc << <gGrid, bGrid >> > (d_img, clusterIDs, clustMean, input.rows, input.cols, clusterSize, minSize);
		}
		else {
			imOut << <gGrid, bGrid >> > (d_img, clusterIDs, clustMean, input.rows, input.cols);
		}
		bool boundaryDraw = true;
		if (boundaryDraw) {
			clusterBoundaryDraw << <gGrid, bGrid >> > (d_img, clusterIDs, input.rows, input.cols, clustMean, diff, false);
		}
		int* outClusters = new int[input.rows * input.cols];
		size = sizeof(unsigned char) * input.rows * input.cols;
		cudaMemcpy( (unsigned char*)&mVec[0], d_img, size, cudaMemcpyDeviceToHost);
		size = sizeof(int) * input.rows * input.cols;
		cudaMemcpy(outClusters, clusterIDs, size, cudaMemcpyDeviceToHost);
		cudaFree(clustMean); cudaFree(clusterIDs); cudaFree(clusterSize); cudaFree(newIDs); cudaFree(d_img);
		Mat retImg = Mat(input.rows, input.cols, CV_8UC1);
		for (int r = 0; r < input.rows; r++) {
			for (int c = 0; c < input.cols; c++) {
				int index = input.cols * r + c;
				 retImg.at<uchar>(r, c) = mVec[index];
			}
		}

		return std::make_pair(retImg, outClusters);
	}

	std::vector<float> meanTest(cv::Mat input) {
		input = Mat(4, 4, CV_8UC1);
		//std::vector<uchar> mVec((input.rows * input.cols), 0);
		std::vector<uchar> mVec(16, 1);
		mVec[5] = 5; mVec[6] = 5; mVec[9] = 5; mVec[10] = 5;
		for (int r = 0; r < input.rows; r++) {
			for (int c = 0; c < input.cols; c++) {
				int index = input.cols * r + c;
				input.at<uchar>(r, c) = mVec[index];
			}
		}
		//if (inputBW.isContinuous()) {
			//mVec.assign(inputBW.data, inputBW.data + inputBW.total());
		//}
		//else {
		for (int r = 0; r < input.rows; r++) {
			for (int c = 0; c < input.cols; c++) {
				int index = input.cols * r + c;
				mVec[index] = input.at<uchar>(r, c);
			}
		}
		//}

		unsigned char* d_img;
		float* clustMean;
		int* clusterSize, * clusterIDs, * newIDs;
		int size = sizeof(unsigned char) * input.rows * input.cols;
		cudaMalloc((void**)&d_img, size);
		cudaMemcpy(d_img, (unsigned char*)&mVec[0], size, cudaMemcpyHostToDevice);
		size = sizeof(int) * input.rows * input.cols;
		cudaMalloc((void**)&clusterSize, size); cudaMalloc((void**)&clusterIDs, size); cudaMalloc((void**)&newIDs, size);
		size = sizeof(float) * input.rows * input.cols;
		cudaMalloc((void**)&clustMean, size);
		std::pair<float, float> imStat = varNOMEM(d_img, input.rows, input.cols);
		//float diff = imStat.first * 0.5;
		float diff = 6;
		printf("var: %f,  mean: %f\n", imStat.first, imStat.second);
		int numBlocksH = input.cols / BLOCK_SIZE;
		if (input.cols % BLOCK_SIZE > 0) {
			numBlocksH++;
		}
		int numBlocksV = input.rows / BLOCK_SIZE;
		if (input.rows % BLOCK_SIZE > 0) {
			numBlocksV++;
		}
		dim3 gGrid(numBlocksV, numBlocksH);
		dim3 bGrid(BLOCK_SIZE, BLOCK_SIZE);

		clusterMeanInit << <gGrid, bGrid >> > (input.rows, input.cols, clustMean, clusterSize, clusterIDs, newIDs, d_img);
		ngbrCompLE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);
		ngbrCompRE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

		

		clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
		clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
		clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
		ngbrCompUE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);
		ngbrCompDE << < gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, diff, newIDs, clustMean);

		

		clusterMeanSum << <gGrid, bGrid >> > (d_img, clusterIDs, clusterSize, input.rows, input.cols, clustMean, newIDs);
		clusterMeanDiv << <gGrid, bGrid >> > (clustMean, input.rows, input.cols, clusterSize);
		clusterSync << <gGrid, bGrid >> > (clusterIDs, input.rows, input.cols, newIDs, clusterSize, clustMean);
		imOut << <gGrid, bGrid >> > (d_img, clusterIDs, clustMean, input.rows, input.cols);

		size = sizeof(unsigned char) * input.rows * input.cols;
		cudaMemcpy((unsigned char*)&mVec[0], d_img, size, cudaMemcpyDeviceToHost);

		cudaFree(clustMean); cudaFree(clusterIDs); cudaFree(clusterSize); cudaFree(newIDs); cudaFree(d_img);

		for (int i = 0; i < input.rows; i++) {
			for (int j = 0; j < input.cols; j++) {
				int index = input.cols * i + j;
				printf("%d, ", mVec[index]);
			}
			printf("\n");
		}
		return Mat();
	}
}
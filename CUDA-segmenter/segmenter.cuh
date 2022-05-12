#pragma once
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace seg {
	float mean(cv::Mat input);

	//variance first, mean second
	cv::Mat laplaceEdge(cv::Mat inputBW);
	std::pair<float, float> variance(cv::Mat input);
	std::pair<cv::Mat, int*> segment(cv::Mat input, float diffParam, int cycles);
	cv::Mat clusterIsolate(cv::Mat input, int* clusterIDS, char R, char G, char B, int clusterNum);
	std::vector<float> meanTest(cv::Mat input);

}
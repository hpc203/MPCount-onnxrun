#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <string>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;


class MPCount
{
public:
	MPCount(string modelpath);
	int detect(Mat frame, Mat& result_map);
private:
	vector<float> input_image;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Single Domain Generalization for Crowd Counting");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	const vector<const char*> input_names = {"input"};
	const vector<const char*> output_names = {"output"};
    const int unit_size=16;
    const int log_para=1000;

    void preprocess(const Mat& frame, Mat& x, int& left, int& top, int& right, int& bottom);
};

MPCount::MPCount(string model_path)
{
	//OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    // std::wstring widestr = std::wstring(model_path.begin(), model_path.end());   ////windows写法
	// ort_session = new Session(env, widestr.c_str(), sessionOptions);           ////windows写法
    ort_session = new Session(env, model_path.c_str(), sessionOptions);          ////linux写法
}

void MPCount::preprocess(const Mat& frame, Mat& x, int& left, int& top, int& right, int& bottom)
{
    const int h = frame.rows;
    const int w = frame.cols;
    int new_h = h;
    int new_w = w;
    if (w % this->unit_size != 0)
    {
        new_w = (floor((float)w / this->unit_size) + 1) * this->unit_size;
    }
    if (h % this->unit_size != 0)
    {
        new_h = (floor((float)h / this->unit_size) + 1) * this->unit_size;
    }

    if (h >= new_h)
    {
        top = 0;
        bottom = 0;
    }       
    else
    {
        int dh = new_h - h;
        top = floor((float)dh / 2);
        bottom = floor((float)dh / 2) + dh % 2;
    }
    if (w >= new_w)
    {
        left = 0;
        right = 0;
    }
    else
    {
        int dw = new_w - w;
        left = floor((float)dw / 2);
        right = floor((float)dw / 2) + dw % 2;
    }
    Mat padded_image; 
    copyMakeBorder(frame, padded_image, top, bottom, left, right, BORDER_CONSTANT, Scalar(0, 0, 0));
    cvtColor(padded_image, x, COLOR_BGR2RGB);
}

int MPCount::detect(Mat frame, Mat& result_map)
{
	Mat x;
    int left, top, right, bottom;
    this->preprocess(frame, x, left, top, right, bottom);
	array<int64_t, 4> input_shape_{ 1, 3, x.rows, x.cols };
    x.convertTo(x, CV_32FC3, 1 / 127.5, -1.0);
    vector<Mat> rgbChannels(3);
    split(x, rgbChannels);
	const int image_area = x.rows * x.cols;
    this->input_image.clear();
	this->input_image.resize(1 * 3 * image_area);
    int single_chn_size = image_area * sizeof(float);
	memcpy(this->input_image.data(), (float *)rgbChannels[0].data, single_chn_size);
    memcpy(this->input_image.data() + image_area, (float *)rgbChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)rgbChannels[2].data, single_chn_size);
	x.release();

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, this->input_image.data(), this->input_image.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	
    std::vector<int64_t> out_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const int outHeight = out_shape[2];
    const int outWidth = out_shape[3];
	float* pred = ort_outputs[0].GetTensorMutableData<float>();
	Mat result(outHeight, outWidth, CV_32FC1, pred);
    Rect crop_roi(left, top, outWidth-right-left, outHeight-top-bottom);
    
    result(crop_roi).copyTo(result_map);
    int people_count = int(float(cv::sum(result_map)[0]) / this->log_para);
    return people_count;
}

Mat draw_result(const Mat& image, const Mat& result_map, const int people_count)
{
    Mat drawimg = result_map.clone();
    double min_value, max_value;
	minMaxLoc(drawimg, &min_value, &max_value, 0, 0);
    drawimg = (drawimg - min_value) / (max_value - min_value + 1e-5);
    drawimg *= 255.0;
    drawimg.convertTo(drawimg, CV_8UC1);

    cv::applyColorMap(drawimg, drawimg, COLORMAP_JET);
    resize(drawimg, drawimg, Size(image.cols, image.rows));
    cv::addWeighted(image, 0.35, drawimg, 0.65, 1.0, drawimg);
    cv::putText(drawimg, "People Count : " + std::to_string(people_count), cv::Point(10, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    return drawimg;
}

int main()
{
	MPCount mynet("/home/wangbo/MPCount/weights/MPCount_qnrf.onnx");  
	string imgpath = "/home/wangbo/MPCount/demo.jpg";
	Mat frame = imread(imgpath);

	Mat result_map;
	int people_count = mynet.detect(frame, result_map);
    Mat drawimg = draw_result(frame, result_map, people_count);
	
	imwrite("result.jpg", drawimg);
    // namedWindow("MPCount Demo : Original Image", WINDOW_NORMAL);
	// imshow("MPCount Demo : Original Image", frame);
	// namedWindow("MPCount Demo : Activation Map", WINDOW_NORMAL);
	// imshow("MPCount Demo : Activation Map", drawimg);
	// waitKey(0);
	// destroyAllWindows();
}
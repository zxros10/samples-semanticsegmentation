/**
 * ============================================================================
 *
 * Copyright (C) 2018, Hisilicon Technologies Co., Ltd. All Rights Reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *   1 Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   2 Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   3 Neither the names of the copyright holders nor the names of the
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * ============================================================================
 */

#include "general_post.h"

#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

#include "hiaiengine/log.h"
#include "opencv2/opencv.hpp"
#include "tool_api.h"

using hiai::Engine;
using namespace std;

namespace {
// callback port (engine port begin with 0)
const uint32_t kSendDataPort = 0;

// sleep interval when queue full (unit:microseconds)
const __useconds_t kSleepInterval = 200000;
const  string LUT_file = "./city.png";


// size of output tensor vector should be 2.
const uint32_t kOutputTensorSize = 1;
const uint32_t kOutputTesnorIndex = 0;

const uint32_t kCategoryIndex = 2;
const uint32_t kScorePrecision = 3;

// bounding box line solid
const uint32_t kLineSolid = 2;

const string kTopNIndexSeparator = ":";
const string kTopNValueSeparator = ",";

// output image prefix
const string kOutputFilePrefix = "out_";

// boundingbox tensor shape
const static std::vector<uint32_t> kDimDetectionOut = {19, 512, 1024};

// opencv draw label params.
const double kFountScale = 0.5;
const cv::Scalar kFontColor(0, 0, 255);
const uint32_t kLabelOffset = 11;
const string kFileSperator = "/";

// opencv color list for boundingbox
const vector<cv::Scalar> kColors {
  cv::Scalar(237, 149, 100), cv::Scalar(0, 215, 255), cv::Scalar(50, 205, 50),
  cv::Scalar(139, 85, 26)};
// output tensor index
enum BBoxIndex {kTopLeftX, kTopLeftY, kLowerRigltX, kLowerRightY, kScore};

}
 // namespace

// register custom data type
HIAI_REGISTER_DATA_TYPE("EngineTrans", EngineTrans);

HIAI_StatusT GeneralPost::Init(
  const hiai::AIConfig &config,
  const vector<hiai::AIModelDescription> &model_desc) {
  // do noting
  return HIAI_OK;
}

bool GeneralPost::SendSentinel() {
  // can not discard when queue full
  HIAI_StatusT hiai_ret = HIAI_OK;
  shared_ptr<string> sentinel_msg(new (nothrow) string);
  do {
    hiai_ret = SendData(kSendDataPort, "string",
                        static_pointer_cast<void>(sentinel_msg));
    // when queue full, sleep
    if (hiai_ret == HIAI_QUEUE_FULL) {
      printf("queue full, sleep 200ms");
      usleep(kSleepInterval);
    }
  } while (hiai_ret == HIAI_QUEUE_FULL);

  // send failed
  if (hiai_ret != HIAI_OK) {
    printf("call SendData failed, err_code=%d", hiai_ret);
    return false;
  }
  return true;
}
string GenerateTopNStr(int32_t top_n, const vector<float> &varr) {
  // if topN max than size, only return size count
  if (top_n > varr.size()) {
    top_n = varr.size();
  }

  // generate index vector from 0 ~ size -1
  vector<size_t> idx(varr.size());
  iota(idx.begin(), idx.end(), 0);
  // sort by original data
  sort(idx.begin(), idx.end(),
       [&varr](size_t i1, size_t i2) {return varr[i1] > varr[i2];});

  // generate result
  stringstream top_stream;
  for (int32_t i = 0; i < top_n; i++) {
    top_stream << idx[i] << kTopNIndexSeparator;
    top_stream << to_string(varr[idx[i]]) << kTopNValueSeparator;
  }

  // return string(need sub last character)
  string result_str = "";
  top_stream >> result_str;
  result_str.pop_back();
  return result_str;
}

HIAI_StatusT GeneralPost::ErfNetPostProcess(
  const shared_ptr<EngineTrans> &result) {

  string file_path = result->image_info.path;

  /*if inference result is null return */
  if (result->inference_res.empty()){  	
    ERROR_LOG("Failed to deal file=%s. Reason: inference result empty.", file_path.c_str());
	return HIAI_ERROR;
  }

  /* read lut file */
  cv::Mat label_colours = cv::imread(LUT_file,1);
  
  printf("label_colours row %d \n",label_colours.rows);
  printf("label_colours col %d \n",label_colours.cols);
  printf("label_colours dims %d \n",label_colours.dims);
  printf("label_colours channels %d \n",label_colours.channels());
 
  /*read inference result */
  Output out2 = result->inference_res[0];
  int32_t size = out2.size / sizeof(float);  
  printf("ErfNetPostProcess size  %d  \n",size);
  
  if (size <= 0) {
    ERROR_LOG("Failed to deal file=%s. Reason: inference result size=%d error.",  file_path.c_str(), size);
    return HIAI_ERROR;
  } 
  /*

  float *res = new(nothrow) float[size];
  if (res == nullptr){
	 ERROR_LOG("Failed to deal file=%s. Reason: new float array failed.",file_path.c_str());
	 return HIAI_ERROR;
  }
  
  errno_t mem_ret = memcpy_s(res, sizeof(float)* size, out2.data.get(),out2.size);
  if(mem_ret != EOK){
  	delete[] res;
	return HIAI_ERROR;
  }
 
  ofstream outfile("out.txt", ios::trunc);  
  for (int i = 0; i < size; i++)
  {
	  outfile  <<res[i]<<  endl;
  };  
  outfile.close();  
  delete[] res; */

   // compute argmax
  cv::Mat class_each_row (19, 1024*512, CV_32FC1, const_cast<float *>((float *)out2.data.get()));
  class_each_row = class_each_row.t(); // transpose to make each row with all probabilities
  cv::Point maxId;	// point [x,y] values for index of max
  double maxValue;	// the holy max value itself
  cv::Mat prediction_map(512,1024,CV_8UC1);
  
  printf("ErfNetPostProcess class_each_row.rows %d  end \n",class_each_row.rows);
  for (int i=0;i<class_each_row.rows;i++){
		minMaxLoc(class_each_row.row(i),0,&maxValue,0,&maxId);	
		prediction_map.at<uchar>(i) = maxId.x;	   
  }
  
  printf("ErfNetPostProcess minMaxLoc  end \n");
  cv::cvtColor(prediction_map.clone(), prediction_map, CV_GRAY2BGR);
  cv::cvtColor(label_colours, label_colours, CV_RGB2BGR);
  cv::Mat output_image,mat;
  cv::LUT(prediction_map, label_colours, output_image);
  //resize(mat,output_image,cv::Size(result->image_info.width, result->image_info.height),1024/result->image_info.width,512/result->image_info.height);      

  printf("output_image row %d \n",output_image.rows);
  printf("output_image col %d \n",output_image.cols);
  printf("output_image dims %d \n",output_image.dims);
  printf("output_image channels %d \n",output_image.channels());

  stringstream sstream;
  int pos = result->image_info.path.find_last_of(kFileSperator);
  string file_name(result->image_info.path.substr(pos + 1));
  bool save_ret(true);
  sstream.str("");
  sstream << result->console_params.output_path << kFileSperator   << kOutputFilePrefix << file_name;
  string output_path = sstream.str();
  
  save_ret = cv::imwrite(output_path, output_image);
  if (!save_ret) {
    ERROR_LOG("Failed to deal file=%s. Reason: save image failed.",
              result->image_info.path.c_str());
    return HIAI_ERROR;
  }

  printf("output_image channels imwrite  end \n");

  return HIAI_OK;  
}

HIAI_IMPL_ENGINE_PROCESS("general_post", GeneralPost, INPUT_SIZE) {
HIAI_StatusT ret = HIAI_OK;

// check arg0
if (arg0 == nullptr) {
  ERROR_LOG("Failed to deal file=nothing. Reason: arg0 is empty.");
  return HIAI_ERROR;
}

// just send to callback function when finished
shared_ptr<EngineTrans> result = static_pointer_cast<EngineTrans>(arg0);
if (result->is_finished) {
  if (SendSentinel()) {
    return HIAI_OK;
  }
  ERROR_LOG("Failed to send finish data. Reason: SendData failed.");
  ERROR_LOG("Please stop this process manually.");
  return HIAI_ERROR;
}

// inference failed
if (result->err_msg.error) {
  ERROR_LOG("%s", result->err_msg.err_msg.c_str());
  return HIAI_ERROR;
}

// arrange result
  return ErfNetPostProcess(result);
}

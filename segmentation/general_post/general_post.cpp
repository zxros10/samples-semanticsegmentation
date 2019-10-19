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


// size of output tensor vector should be 2.
const uint32_t kOutputTensorSize = 1;
const uint32_t kOutputTesnorIndex = 0;

const uint32_t kCategoryIndex = 2;
const uint32_t kScorePrecision = 3;

// bounding box line solid
const uint32_t kLineSolid = 2;

// Max categories Num 
const uint32_t MAX_NUM = 256;

const string kTopNIndexSeparator = ":";
const string kTopNValueSeparator = ",";

// output image prefix
const string kOutputFilePrefix = "out_";


// opencv draw label params.
const string kFileSperator = "/";


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
      HIAI_ENGINE_LOG("queue full, sleep 200ms");
      usleep(kSleepInterval);
    }
  } while (hiai_ret == HIAI_QUEUE_FULL);

  // send failed
  if (hiai_ret != HIAI_OK) {
    HIAI_ENGINE_LOG(HIAI_ENGINE_RUN_ARGS_NOT_RIGHT,
                    "call SendData failed, err_code=%d", hiai_ret);
    return false;
  }
  return true;
}

HIAI_StatusT GeneralPost::SegmentationNetPostProcess(
  const shared_ptr<EngineTrans> &result) {

  string file_path = result->image_info.path;

  /*if inference result is null return */
  if (result->inference_res.empty()){  	
    ERROR_LOG("Failed to deal file=%s. Reason: inference result empty.", file_path.c_str());
	return HIAI_ERROR;
  }

  /* judge categories num  */
  if(MAX_NUM < result->console_params.output_nums)
  {  
    ERROR_LOG("Failed to deal categories num  %d is large than 255.", result->console_params.output_nums);
	return HIAI_ERROR;
  }

  /* set lut file */  
  uchar lutData[MAX_NUM * 3];
  uchar label;
  for (int i = 0; i<MAX_NUM; i++)	
  {
	  int j =0;
	  label = i;
	  while(label >0)
	  {
		lutData[i * 3]  |= (((label >> 0) & 1) << (7 - j));			
		lutData[i * 3 + 1] |= (((label >> 1) & 1) << (7 - j));			
		lutData[i * 3 + 2] |= (((label >> 2) & 1) << (7 - j));	
		label >>=3;
		j += 1;
	  }	
  }		  
  
  cv::Mat label_colours(1, MAX_NUM, CV_8UC3, lutData);
  cv::cvtColor(label_colours, label_colours, CV_RGB2BGR);
  
  /*read inference result */
  Output out = result->inference_res[0];
  int32_t size = out.size / sizeof(float);  
  
  int32_t ModelOutSize =  result->console_params.model_width* result->console_params.model_height*result->console_params.output_nums; 
  INFO_LOG("SegmentationNetPostProcess size  %d  \n",size);
  
  if ((size <= 0)||(size < ModelOutSize)) {
    ERROR_LOG("Failed to deal file=%s. Reason: inference result size=%d error.",  file_path.c_str(), size);
    return HIAI_ERROR;
  }
  
  /*copy inference data */
  cv::Mat class_each_row (result->console_params.output_nums,  result->console_params.model_width* result->console_params.model_height, CV_32FC1, const_cast<float *>((float *)out.data.get()));

  /* transpose to make each row with all probabilities */
  class_each_row = class_each_row.t(); 
  cv::Point maxId;	// point [x,y] values for index of max
  double maxValue;	// the holy max value itself
  cv::Mat prediction_map(result->console_params.model_height, result->console_params.model_width, CV_8UC1);
  
  /* select each pixle max */
  for (int i = 0; i<class_each_row.rows;i++){
  	minMaxLoc(class_each_row.row(i),0,&maxValue,0,&maxId);	
	prediction_map.at<uchar>(i) = maxId.x;		
  }
  
  INFO_LOG("each rows %d  \n",class_each_row.rows);    
  INFO_LOG(" minMaxLoc  end \n");
  
  /* get inference out pic*/
  cv::cvtColor(prediction_map.clone(), prediction_map, CV_GRAY2BGR);  
  cv::Mat Infer_output_image	;
  cv::LUT(prediction_map, label_colours, Infer_output_image);

  /* get origin pic */
  cv::Mat originImage= cv::imread(result->console_params.input_path, 1); 
  resize(originImage,originImage,cv::Size(result->console_params.model_width,result->console_params.model_height),
  	originImage.cols/result->console_params.model_width,originImage.rows/result->console_params.model_height);	

  /* get display image*/
  cv::Mat displayImage	;  
  displayImage = 0.5*Infer_output_image +0.5*originImage;
  
  /*set output img name */
  stringstream sstream;
  int pos = result->image_info.path.find_last_of(kFileSperator);
  string file_name(result->image_info.path.substr(pos + 1));
  
  bool save_ret(true);
  sstream.str("");
  sstream << result->console_params.output_path << kFileSperator << kOutputFilePrefix << file_name;
  string output_path = sstream.str();

  /*write img */ 
  save_ret = cv::imwrite(output_path, displayImage);
  if (!save_ret) {
    ERROR_LOG("Failed to deal file=%s. Reason: save image failed.",
              result->image_info.path.c_str());
    return HIAI_ERROR;
  }

  INFO_LOG("output_image channels imwrite  end \n");

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
  return SegmentationNetPostProcess(result);
}

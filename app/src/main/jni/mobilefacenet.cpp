// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#include <jni.h>
#include "mobileface.id.h"
#include<fstream>
#include<iostream>

#include <string>
#include <vector>

// ncnn
#include "net.h"
#include "benchmark.h"

#include "mobilefacenet.h"
#define LOG(...) __android_log_print(ANDROID_LOG_INFO, "native-lib", __VA_ARGS__)
#define THRESHOLD 70.36f



static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static std::vector<std::string> squeezenet_words;
static ncnn::Net mobilefacenet;

static double duration;
static  jboolean if_gpu;

using namespace std;
static std::vector<std::string> split_string(const std::string& str, const std::string& delimiter)
{
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "API", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "API", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL
Java_com_tencent_mobilefacenet_API_Init(JNIEnv *env, jobject thiz, jobject assetManager)
{
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    mobilefacenet.opt = opt;

    // init param
    {
        int ret = mobilefacenet.load_param_bin(mgr, "mobileface.param.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "API", "load_param_bin failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        int ret = mobilefacenet.load_model(mgr, "mobileface.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "API", "load_model failed");
            return JNI_FALSE;
        }
    }

    duration=.0f;

    return JNI_TRUE;
}

void dot(float *a, float *b, float &result, int len) {
    result=.0f;
    for(int i=0;i<len;i++){
        result+=a[i] * b[i] ;
    }
}

void clip(float &a) {
    if(a<-1.0f){
        a=-1.0f;
    }else if(a>1.0f){
        a=1.0f;
    }
}
void get_output(cv::Mat &img, float *a, bool flip) {
    cv::Mat img2= img;
    if (flip)
    {
        cv::flip(img, img2, 1);
    }
    // 把opencv的mat转换成ncnn的mat
    ncnn::Mat input = ncnn::Mat::from_pixels(img2.data, ncnn::Mat::PIXEL_RGBA2BGR, img2.cols, img2.rows);

    const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};

    input.substract_mean_normalize(mean_vals, norm_vals);
    // pretty_print(input);

    // clock_t start,finish;
    // start=clock();
    auto t1 = std::chrono::high_resolution_clock::now();
    // ncnn前向计算
    ncnn::Extractor extractor = mobilefacenet.create_extractor();
    extractor.set_vulkan_compute(if_gpu);
    extractor.input(mobileface_param_id::BLOB_input, input);
    ncnn::Mat output;
    extractor.extract(mobileface_param_id::BLOB_output, output);

    std::chrono::duration<double, std::milli> fp_ms;
    auto t2 = std::chrono::high_resolution_clock::now();
    fp_ms = t2 - t1;
    // auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(fp_ms);
    duration+=fp_ms.count();

    float value=.0f;
    const float* ptr = output.channel(0);
    // float a[128];
    for (int y = 0; y < output.h; y++)
    {
        for (int x = 0; x < output.w; x++)
        {
            a[x]=ptr[x];
        }
        ptr += output.w;
    }
}

void get_feature(cv::Mat &image,  float *a) {
    float x1[128],x2[128];
    get_output(image,x1,false);
    get_output(image,x2,true);

    float value=.0f;
    for (int x = 0; x < 128; x++)
    {
        a[x]=x1[x]+x2[x];
        value+=a[x]*a[x];
    }
    value = sqrt(value);
    for (int x = 0; x < 128; x++) {
        a[x]/=value;
    }
}


string compare(JNIEnv *env,cv::Mat &image1,cv::Mat &image2,int &answer){
    float img1[128];
    float img2[128];
    get_feature(image1,img1);
    get_feature(image2,img2);
    float result;
    dot(img1,img2,result,128);
    clip(result);
    result = acos(result) * 180.0f / 3.1415926f;
    const char* r_ = std::to_string(result).c_str();
    LOG("角度：%s", r_);
    std::string result_str;

    if(result<THRESHOLD)
        result_str="the same";
    else
        result_str="different";
//    jstring r = env->NewStringUTF(result_str.c_str());
    return result_str;
};

void BitmapToMat2(JNIEnv *env, jobject& bitmap, cv::Mat& mat, jboolean needUnPremultiplyAlpha)
{
    AndroidBitmapInfo info;
    void *pixels = 0;
    cv::Mat &dst = mat;

    CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
    CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
              info.format == ANDROID_BITMAP_FORMAT_RGB_565);
    CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
    CV_Assert(pixels);

    dst.create(info.height, info.width, CV_8UC4);
    cv::Mat tmp(info.height, info.width, CV_8UC4, pixels);

    tmp.copyTo(dst);

//    std::string r="";
//    for (int i = 0; i < info.height; i++) {
//        for (int j = 0; j < info.width; j++) {
//            int index = i * info.width + j;
//            int b1 = (int) dst.data[4 * index + 0];
//            int g1 = (int) dst.data[4 * index + 1];
//            int r1 = (int) dst.data[4 * index + 2];
//            int a = (int) dst.data[4 * index + 3];
//            r = r + std::to_string(b1) + " " + std::to_string(g1) + " " + std::to_string(r1)+" " ;
//        }
//    }
//    const char* r_ = r.c_str();
//    LOG("%s", r_);

    AndroidBitmap_unlockPixels(env, bitmap);

}

void BitmapToMat(JNIEnv *env, jobject& bitmap, cv::Mat& mat) {
    BitmapToMat2(env, bitmap, mat, false);
}

// public native String Detect(Bitmap bitmap, boolean use_gpu);
JNIEXPORT jstring JNICALL
Java_com_tencent_mobilefacenet_API_Detect(JNIEnv *env, jobject thiz, jobject bitmap1,
                                          jobject bitmap2, jboolean use_gpu)
{
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
    {
        return env->NewStringUTF("no vulkan capable gpu");
    }
    if_gpu=use_gpu;
    duration=.0f;

//    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap1, &info);
    int width = info.width;
    int height = info.height;
    if (width != 112 || height != 112)
        return NULL;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    cv::Mat mat_1;
    cv::Mat mat_2;
    int answer=0;
    BitmapToMat(env, bitmap1, mat_1);
    BitmapToMat(env, bitmap2, mat_2);
    string r=compare(env,mat_1,mat_2,answer);
//    duration = ncnn::get_current_time()-start_time;
//    LOG("耗时：%.2f",duration);
    string  duration_str=to_string(duration);
    string result_str=r+"-"+duration_str+"ms";
    jstring result = env->NewStringUTF(result_str.c_str());

    return result;

}

}

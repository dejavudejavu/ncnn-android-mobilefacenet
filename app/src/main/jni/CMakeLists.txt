project(mobilefacenet)

cmake_minimum_required(VERSION 3.10)

# 配置OpenCV
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-2.4.13.7-android/sdk/native/jni)
find_package(OpenCV REQUIRED)


# 配置ncnn
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

#配置log
find_library(log-lib log)


add_library(mobilefacenet SHARED mobilefacenet.cpp)

target_link_libraries(mobilefacenet ${log-lib} ncnn ${OpenCV_LIBS})

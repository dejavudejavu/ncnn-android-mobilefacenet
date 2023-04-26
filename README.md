# ncnn-android-mobilefacenet

A face recognition app

this is a sample ncnn android project, it depends on ncnn library only

https://github.com/Tencent/ncnn

## how to build and run
### step1
https://github.com/Tencent/ncnn/releases

* Download ncnn-YYYYMMDD-android-vulkan.zip or build ncnn for android yourself
* Extract ncnn-YYYYMMDD-android-vulkan.zip into **app/src/main/jni** and change the **ncnn_DIR** path to yours in **app/src/main/jni/CMakeLists.txt**

### step2
* Open this project with Android Studio, build it and enjoy!
* 待检测的人脸图片需要裁剪成方形，并且尽量包含整个面部，不含其他物体， 例：<br>
![相同（cpu）](./imgs/example.jpg#pic_center)

## screenshot
检测相同的图片,GPU耗时比CPU更长,根据[官方解释](https://github.com/Tencent/ncnn/wiki/FAQ-ncnn-vulkan),这是因为很多针对GPU的优化还没有完成（例如winograd卷积，算子融合，fp16存储和算术等），而且arm架构下的CPU优化已经做够充分，所以CPU下更快。

相同（cpu）:<br> 
![相同（cpu）](./imgs/cpu_same.jpg#pic_center)<br>
相同（gpu）:<br> 
![相同（gpu）](./imgs/gpu_same.jpg#pic_center)<br>
不同（cpu）:<br> 
![不同（cpu）](./imgs/cpu_different.jpg#pic_center)<br>


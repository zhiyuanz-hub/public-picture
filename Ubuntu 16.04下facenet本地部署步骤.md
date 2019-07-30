# Ubuntu 16.04下facenet本地部署步骤
## 1. 安装Anaconda
+ 首先从官网上下载安装包(以 Anaconda3-5.3.1-Linux-x86_64.sh 为例)，然后：sudo bash Anaconda3-5.3.1-Linux-x86_64.sh
+ 添加Anaconda环境变量
1. 终端输入: gedit ~/.bashrc
2. 添加到文件最后：export PATH=/home/自己的用户名/anaconda3/bin:$PATH
3. 立刻加载修改后的设置，使之生效： source ~/.bashrc
## 2. 下载Facenet源码文件
git clone https://github.com/davidsandberg/facenet.git
## 3. 下载lfw数据集，并解压
下载链接： http://vis-www.cs.umass.edu/lfw/lfw.tgz
## 4. 下载预先训练的模型
下载链接: https://github.com/davidsandberg/facenet，下载文件名为 20180402-114759的这个并解压，这个文件的下载地址为： https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-（需要翻墙用Google下载），得到的文件夹里有4个文件
## 5. 环境配置
打开 facenet 这个文件夹，找到requirements.txt文件并打开，需要的环境有：tensorflow==1.7、scipy、scikit-learn、opencv-python、h5py、matplotlib、Pillow、requests、psutil
+ 可以直接执行：sudo pip install -r requirements.txt 进行全部下载
+ 也可以先创建一个tf环境，比如py36tf，conda create -n py36tf python=3.6，然后进入这
个环境：conda activate py36tf,最后使用pip install tensorflow==1.7 scipy==1.1.0 scikit-learn opencv-python h5py matplotlib Pillow requests psutil numpy==1.16.1将所需要的环境进行全部安装（这里进行配置的时候将numpy的版本设置为1.16.1，scipy的版本设置为1.1.0，以防止后续操作出现一些问题）
## 6. 预处理图片:人脸检测和人脸对齐
在facenet文件夹里执行：python src/align/align_dataset_mtcnn.py ../lfw ../lfw_align_160 --image_size 160 --margin 32 --random_order

出现错误:
Traceback (most recent call last):
  File "src/align/align_dataset_mtcnn.py", line 34, in <module>
    import facenet
ImportError: No module named 'facenet'

解决办法:将src/文件夹路径添加到PYTHONPATH环境变量下：export PYTHONPATH=$PATH:/home/用户名/facenet/src

再次执行：python src/align/align_dataset_mtcnn.py ../lfw ../lfw_align_160 --image_size 160 --margin 32 --random_order

等待执行完毕后可以看到最后两行显示:
Total number of images: 13233
Number of successfully aligned images: 13233
## 7. 在LFW数据库中验证预先训练的模型
在facenet下执行：python src/validate_on_lfw.py ../lfw_align_160/ ../20180402-114759
等待执行完毕后结果输出为：
2019-07-30 16:25:15.574038: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Model directory: ../20180402-114759
Metagraph file: model-20180402-114759.meta
Checkpoint file: model-20180402-114759.ckpt-275
Runnning forward pass on LFW images
............
Accuracy: 0.98500+-0.00658
Validation rate: 0.90067+-0.02403 @ FAR=0.00067
Area Under Curve (AUC): 0.998
Equal Error Rate (EER): 0.016
2019-07-30 16:38:07.226974: W tensorflow/core/kernels/queue_base.cc:277] _2_input_producer: Skipping cancelled enqueue attempt with queue not closed
2019-07-30 16:38:07.227885: W tensorflow/core/kernels/queue_base.cc:285] _1_FIFOQueueV2: Skipping cancelled dequeue attempt with queue not closed
2019-07-30 16:38:07.227931: W tensorflow/core/kernels/queue_base.cc:285] _1_FIFOQueueV2: Skipping cancelled dequeue attempt with queue not closed
2019-07-30 16:38:07.228035: W tensorflow/core/kernels/queue_base.cc:285] _1_FIFOQueueV2: Skipping cancelled dequeue attempt with queue not closed
2019-07-30 16:38:07.228052: W tensorflow/core/kernels/queue_base.cc:285] _1_FIFOQueueV2: Skipping cancelled dequeue attempt with queue not closed
2019-07-30 16:38:07.228062: W tensorflow/core/kernels/queue_base.cc:285] _1_FIFOQueueV2: Skipping cancelled dequeue attempt with queue not closed
2019-07-30 16:38:07.228071: W tensorflow/core/kernels/queue_base.cc:285] _1_FIFOQueueV2: Skipping cancelled dequeue attempt with queue not closed
2019-07-30 16:38:07.228081: W tensorflow/core/kernels/queue_base.cc:285] _1_FIFOQueueV2: Skipping cancelled dequeue attempt with queue not closed
2019-07-30 16:38:07.228091: W tensorflow/core/kernels/queue_base.cc:285] _1_FIFOQueueV2: Skipping cancelled dequeue attempt with queue not closed
这个warnings我在facenet官方上找到的答案是：These are warnings that are harmless but a bit annoying. They come from closing the session (which was not done before) but I didn't find a way to make them disappear. I think you can safely ignore them for now.大致意思可以忽略这个warnings.
## 8. 在自己的数据集上应用
新建一个文件夹test_images，里面放三张图片0.jpg、1.jpg、2.jpg，其中0.jpg(赵丽颖)、1.jpg(刘亦菲)、2.jpg(刘亦菲)
在facenet下运行: python src/compare.py ../20180402-114759 ../test_images/0.jpg ../test_images/1.jpg ../test_images/2.jpg
等待执行完毕后结果输出为：
Creating networks and loading parameters
2019-07-30 16:54:15.047314: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Model directory: ../20180402-114759
Metagraph file: model-20180402-114759.meta
Checkpoint file: model-20180402-114759.ckpt-275
Images:
0: ../test_images/0.jpg
1: ../test_images/1.jpg
2: ../test_images/2.jpg
Distance matrix
        0         1         2     
0    0.0000    0.9406    1.0065  
1    0.9406    0.0000    0.6207  
2    1.0065    0.6207    0.0000 







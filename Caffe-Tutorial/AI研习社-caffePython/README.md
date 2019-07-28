# Caffe for Python
##This is a Caffe for python tutorial in Windows.

#Step:
##1、安装Anaconda2；
##2、将pycaffe文件下的所有内容复制到Anaconda2(Python)的"site-##packages"文件下；
##3、在cmd中执行
    pip install protobuf
##4、下载"bvlc_reference_caffenet.caffemodel"，并将其拷贝到caffe的根目录下的"caffe_root/models/bvlc_reference_caffenet/"文件下；
##5、下载"synset_words.txt"，并将其拷贝到caffe的根目录下的"caffe_root/ilsvrc12/'"文件下。
##6、将"demoPython.py"文件中的第19行
    caffe_root = '...'
##改为你自己的caffe根目录即可，例如我的是
    caffe_root = 'H:\\Caffe\\Caffe_CPU_ONLY(Debug_Release)\\caffe-windows\\'  # this file should be run from {caffe_root}/examples (otherwise change this line)
##7、直接双击"demoPython.bat"即可运行！


#环境：Win10 64bit + Python2.7.11.
#*翻译自：[00-classification](http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb)*

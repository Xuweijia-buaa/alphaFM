GCCVERSIONLT8 = $(shell expr `$(CXX) -dumpversion | cut -f1 -d.` \< 8)
CFLAGS = -O3 -w

ifeq "$(GCCVERSIONLT8)" "1"
	CFLAGS += -std=c++11
endif

all:
    # 代码生成可执行文件，到bin目录下。
    # 对fm_train: 该可执行文件不变.  入参是新旧模型路径。每次执行，pipe输入新的数据，作为标准输入，输出新的模型参数文件
    # 本地：
    # model对应的模型文件可以在本地。不一定在hdfs上。只有数据需要放hdfs。
    # 模型文件本身放在hdfs上，
    # 首次训练只需要指定首次输出的模型参数路径： cat test.txt  | ./fm_train -m old_model.txt
    # 增量训练,指定原来的旧模型。输出新的模型参数： cat test2.txt | ./fm_train -im old_model.txt -m new_model.txt
    #              (也可以是文件夹内所有样本，cat的结果是所有样本)：cat data/*  | ./fm_train -m old_model_all.txt
    # hadoop上：
    # hadoop fs -cat test.txt | ./fm_train -m old_model.txt
	$(CXX) $(CFLAGS) fm_train.cpp src/Frame/pc_frame.cpp src/Utils/utils.cpp -I . -o bin/fm_train -lpthread
	$(CXX) $(CFLAGS) fm_predict.cpp src/Frame/pc_frame.cpp src/Utils/utils.cpp -I . -o bin/fm_predict -lpthread
	$(CXX) $(CFLAGS) model_bin_tool.cpp src/Utils/utils.cpp -I . -o bin/model_bin_tool


# 手机银行：把生成的模型文件放到ECS，并且通知排序服务。排序服务close，释放原来的模型（释放堆外内存），从ECS load新模型，完成模型的热更新。


# 提交一个作业，到hadoop.需调用hadoop stream:
# 并指定hdfs上的输入数据，输出数据位置: 把文件转为标准输入，并把标准输出转为文件。
# -mapper是可执行文件位置
# -file是需要上传到集群的文件，含可执行文件
# hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-3.2.1.jar \
#     -input /my_inputs/data/test.txt \
#     -output /fm_outs \
#     -mapper ./fm_train \
# 	-reducer ./fm_train \
#     -file ./fm_train \
#     #-reducer /bin/cat \
# 	#-m model.txt
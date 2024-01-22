1. 代码文件说明
   数据集类文件 code/utils/TSDataSet.py
   模型结构类文件 code/utils/TSCNN.py
   训练文件 code/train.py
   测试文件 code/test.py

2. shell 脚本文件说明
   unzip_data.sh 负责将 data 下的压缩文件进行解压
   repes_install.sh 负责安装依赖 和 创建所需文件夹
   start_train.sh 负责启动训练并保存模型参数
   start_test.sh 负责使用已训练好的模型对测试数据进行预测并压缩保存，压缩文件保存至 code/submit

3. 提交测试结果说明
   提交的测试数据集结果保存在 submit 中，该结果测评分数为 0.2908，被测试模型的权重保存在 code/model 中

4. 训练环境
   pytorch 版本 1.13.0+cu116
   CUDA 版本 11.6
   cuDNN 版本 8302
   其他 python 第三方库请看 requirements.txt

5. 由于训练时出了点意外，本来提交的最高测评结果 (0.2915) 对应的模型参数被覆盖掉了，
   所以提交测评为 0.2908 对应模型参数。
   0.2915 对应模型 A 的训练超参数为 30 epochs，学习率0.00002，使用 CosineAnnealingLR 对学习率进行调整
   0.2908 对应模型是在 模型 A 基础上，继承 A 训练时最后一个epoch学习率，不改变该学习率再训练 30 个epoch的结果

6. 配置并开始训练
   
   ```bash
   bash unzip_data.sh
   bash repes_install.sh
   bash start_train.sh
   ```

7. 测试
   
   ```bash
   bash start_test.sh
   ```
8. 权重参数可以从这个[链接](https://drive.google.com/file/d/1LuNIhWKB1VLwPFpaOq8XquF3ozd7kXyG/view?usp=sharing)下载。
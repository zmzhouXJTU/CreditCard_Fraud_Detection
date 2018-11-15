## 信用卡欺诈检测

使用`Logstic Regression`对信用卡欺诈检测进行分类

### 步骤以及一些需要注意的点

* 特征工程
* 样本不均衡问题的解决(**降采样以及过采样两种方式**)
* 下采样策略
* 交叉验证(充分利用数据，使模型更具说服力)
* 模型评估方法(分类准确率，精确率，召回率，F1值)
* 正则化惩罚(防止模型过拟合，引入**L2正则化**)
* 逻辑回归阈值对结果的影响(通过混淆矩阵的可视化以及召回率来体现)
* 过采样策略(**SMOTE算法**)


### 如何运行？

* 信用卡数据集为"creditcard.csv"，地址为:https://myblogs-photos-1256941622.cos.ap-chengdu.myqcloud.com/%E4%BF%A1%E7%94%A8%E5%8D%A1%E6%AC%BA%E8%AF%88%E6%A3%80%E6%B5%8B/creditcard.csv (可根据需要下载)
* 在和数据集的同一存储目录下运行` python Creditcard_fraud_detection.py`文件即可

### 配置环境

* Python 3.6.5
* numpy 1.15.4
* pandas 0.23.4
* matplotlib 3.0.2
* scikit-learn 0.20.0
* imbalanced-learn 0.4.3
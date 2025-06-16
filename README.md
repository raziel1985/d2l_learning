# 深度学习学习项目 (D2L Learning)

本项目是基于《动手学深度学习》(Dive into Deep Learning) 的学习实践项目，包含了深度学习各个重要领域的代码实现和示例。

## 学习参考资源

本项目配套的官方学习资源：

- 📚 **在线教材**：[《动手学深度学习》](https://zh.d2l.ai/) - 完整的理论知识和代码示例
- 🎓 **官方课程**：[动手学深度学习课程](https://courses.d2l.ai/zh-v2/) - 系统化的课程安排和学习路径
- 🎬 **视频教程**：[李沐老师B站课程](https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497) - 详细的视频讲解
- 💻 **官方代码**：[GitHub仓库](https://github.com/d2l-ai/d2l-zh) - 官方完整代码实现


## 项目结构

### 📊 线性网络 (linear-networks/)
线性模型的基础实现，深度学习的起点。
- `linear-regression-concise.py` - 使用高级API实现线性回归
- `linear-regression-scratch.py` - 从零开始实现线性回归

**核心知识点：**
- 线性回归的数学原理和损失函数
- 梯度下降优化算法
- 正规方程求解
- 特征工程和数据预处理
- 模型评估指标（MSE、MAE等）

### 🎯 Softmax回归 (softmax-regression/)
多分类问题的基础模型实现。
- `image-classification-dataset.py` - 图像分类数据集处理
- `softmax-regression-concise.py` - 使用高级API实现Softmax回归
- `softmax-regression-scratch.py` - 从零开始实现Softmax回归

**核心知识点：**
- Softmax函数的数学原理
- 交叉熵损失函数
- 多分类问题的概率解释
- 独热编码（One-hot Encoding）
- 分类准确率和混淆矩阵

### 🧠 多层感知机 (multilayer-perceptrons/)
深度神经网络的基础构建块。
- `mlp-concise.py` / `mlp-scratch.py` - 多层感知机的两种实现方式
- `dropout.py` - Dropout正则化技术
- `weight-decay.py` - 权重衰减正则化
- `underfit-overfit.py` - 欠拟合和过拟合分析
- `kaggle-hourse-price.py` - Kaggle房价预测竞赛实践
- `common.py` - 通用工具函数

**核心知识点：**
- 多层神经网络架构设计
- 反向传播算法原理
- 激活函数（ReLU、Sigmoid、Tanh）
- 正则化技术（Dropout、权重衰减）
- 偏差-方差权衡
- 模型容量和复杂度控制

### 🔍 卷积神经网络 (convolutional-neural-networks/)
CNN的基础概念和实现。
- `conv-layer.py` - 卷积层的实现
- `padding-and-strides.py` - 填充和步幅机制
- `channels.py` - 多通道处理
- `pooling.py` - 池化层实现
- `lenet.py` - LeNet经典网络架构
- `common.py` - CNN相关工具函数

**核心知识点：**
- 卷积运算的数学原理
- 特征图和感受野概念
- 参数共享和平移不变性
- 池化操作（最大池化、平均池化）
- 卷积层、池化层、全连接层组合
- 图像特征提取和层次化表示

### 🏗️ 现代卷积架构 (convolutional-modern/)
经典和现代的CNN架构实现。
- `alexnet.py` - AlexNet网络架构
- `vgg.py` - VGG网络架构
- `nin.py` - Network in Network架构
- `googlenet.py` - GoogLeNet/Inception架构
- `resnet.py` - ResNet残差网络
- `batch-norm.py` - 批量归一化技术
- `common.py` - 现代CNN架构工具函数

**核心知识点：**
- 深度网络的梯度消失问题
- 残差连接（Skip Connection）
- 批量归一化（Batch Normalization）
- Inception模块和多尺度特征
- 1×1卷积和通道降维
- 网络深度vs宽度的权衡
- 模型压缩和效率优化

### 💻 深度学习计算 (deep-learning-computation/)
深度学习框架的底层计算和优化。
- `model-construction.py` - 模型构建技术
- `parameters.py` - 参数管理和初始化
- `custom-layer.py` - 自定义层的实现
- `read-write.py` - 模型的保存和加载
- `gpu.py` - GPU计算和优化

**核心知识点：**
- 计算图和自动微分
- 参数初始化策略（Xavier、He初始化）
- 模型序列化和反序列化
- GPU并行计算原理
- 内存管理和计算效率
- 自定义层和操作符实现
- 模型部署和推理优化

### 🔄 循环神经网络 (recurrent-neural-networks/)
处理序列数据的基础RNN实现。
- `rnn-concise.py` / `rnn-scratch.py` - RNN的两种实现方式
- `language-models-and-dataset.py` - 语言模型和数据集
- `text-preprocessing.py` - 文本预处理技术
- `sequence.py` - 序列数据处理
- `common.py` - RNN相关工具函数

**核心知识点：**
- 循环神经网络的时间展开
- 梯度爆炸和梯度消失问题
- 通过时间的反向传播（BPTT）
- 语言模型和困惑度
- 序列到序列学习
- 文本分词和词汇表构建
- n-gram模型vs神经语言模型

### 🚀 现代循环架构 (recurrent-modern/)
先进的循环神经网络架构。
- `lstm.py` - 长短期记忆网络
- `gru.py` - 门控循环单元
- `deep-rnn.py` - 深度循环网络
- `bi-rnn.py` - 双向循环网络
- `encoder-decoder.py` - 编码器-解码器架构
- `seq2seq.py` - 序列到序列模型
- `machine-translation-and-dataset.py` - 机器翻译数据集
- `common.py` - 现代RNN架构工具函数

**核心知识点：**
- LSTM的门控机制（遗忘门、输入门、输出门）
- GRU的简化门控结构
- 长期依赖问题的解决方案
- 双向RNN和上下文信息
- 编码器-解码器框架
- 注意力机制的引入
- 机器翻译评估指标（BLEU分数）

### 🎯 注意力机制 (attention-mechanisms/)
注意力机制和Transformer架构。
- `nadaraya-waston.py` - Nadaraya-Watson核回归
- `attention-scoring-functions.py` - 注意力评分函数
- `bahdanau-attention.py` - Bahdanau注意力机制
- `self-attention-and-positional-encoding.py` - 自注意力和位置编码
- `transformer.py` - Transformer架构实现
- `common.py` - 注意力机制工具函数

**核心知识点：**
- 注意力机制的数学原理
- 查询、键、值（Query-Key-Value）机制
- 加性注意力vs乘性注意力
- 自注意力和多头注意力
- 位置编码和序列建模
- Transformer架构详解
- 并行化计算优势

### 🖼️ 计算机视觉 (computer-vision/)
计算机视觉的高级技术和应用。
- `image-augmentation.py` - 图像数据增强
- `fine-tuning.py` - 模型微调技术
- `object-detection-dataset.py` - 目标检测数据集
- `anchor.py` - 锚框生成
- `bounding-box.py` - 边界框处理
- `multiscale-object-detection.py` - 多尺度目标检测
- `ssd.py` - SSD目标检测算法
- `semantic-segmentation-and-dataset.py` - 语义分割
- `fcn.py` - 全卷积网络
- `transposed-conv.py` - 转置卷积
- `neural-style.py` - 神经风格迁移
- `kaggle-cifar10.py` / `kaggle-dog.py` - Kaggle竞赛实践
- `common.py` - 计算机视觉工具函数

**核心知识点：**
- 图像分类、目标检测、语义分割任务
- 数据增强技术和正则化
- 迁移学习和预训练模型
- 锚框机制和非极大值抑制
- 多尺度特征金字塔
- 全卷积网络和上采样
- 损失函数设计（分类损失、回归损失）
- 风格迁移和内容-风格分离

### 🔧 优化算法 (optimization/)
深度学习中的各种优化算法。
- `optimization_intro.py` - 优化算法介绍
- `convexity.py` - 凸性分析
- `gd.py` - 梯度下降算法
- `sgd.py` - 随机梯度下降
- `minibatch-sgd.py` - 小批量随机梯度下降
- `momentum.py` - 动量优化算法
- `adam.py` - Adam优化算法
- `common.py` - 优化算法工具函数

**核心知识点：**
- 凸优化vs非凸优化
- 学习率调度策略
- 动量方法和加速梯度下降
- 自适应学习率算法（AdaGrad、RMSprop、Adam）
- 批量大小对收敛的影响
- 局部最小值和鞍点问题
- 优化算法的理论分析和收敛性

### 🚄 计算性能 (computational-performance/)
深度学习的计算性能优化。
- `multiple-gpus.py` - 多GPU训练基础实现
- `multiple-gpus-concise.py` - 多GPU训练高级API实现

**核心知识点：**
- 数据并行vs模型并行
- GPU内存管理和优化
- 分布式训练策略
- 通信开销和同步机制
- 混合精度训练
- 模型量化和剪枝
- 推理加速技术

### 🗣️ 语言处理预训练 (language-processing-pretraining/)
自然语言处理的预训练模型。
- `bert.py` - BERT模型实现
- `natural-language-inference-bert.py` - 基于BERT的自然语言推理
- `common.py` - NLP预训练模型工具函数

**核心知识点：**
- 预训练-微调范式
- 掩码语言模型（MLM）
- 下一句预测（NSP）任务
- WordPiece分词技术
- 双向编码器架构
- 迁移学习在NLP中的应用
- 自然语言推理任务设计

### 🖼️ 图像资源 (img/)
项目中使用的示例图像文件。
- `autumn-oak.jpg` - 秋季橡树图像
- `banana.jpg` - 香蕉图像
- `cat1.jpg` - 猫咪图像
- `catdog.jpg` - 猫狗组合图像
- `rainier.jpg` - 雷尼尔山图像

## 使用说明

1. **环境要求**：确保安装了PyTorch、NumPy、Matplotlib等必要的深度学习库
2. **学习路径**：建议按照目录顺序学习，从线性网络开始，逐步深入到更复杂的架构
3. **代码风格**：每个主题都提供了"从零开始"和"简洁实现"两种版本，便于理解底层原理和实际应用
4. **实践项目**：包含多个Kaggle竞赛实践，可以检验学习效果

## 学习建议

- 🔰 **初学者**：从`linear-networks`和`softmax-regression`开始
- 🎓 **进阶者**：重点关注`attention-mechanisms`和`computer-vision`
- 🚀 **高级用户**：深入研究`computational-performance`和优化技术

---

本项目旨在通过实践加深对深度学习理论的理解，每个模块都包含详细的代码注释和实现说明。
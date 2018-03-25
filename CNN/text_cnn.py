import tensorflow as tf
import numpy as np
import t_wiki_process



### 定义计算余弦距离（即余弦相似度）
def compute_cosine_distance(x, y):
    with tf.name_scope('cosine_distance'):
        # 公式： cosine=x*y/(|x||y|)
        # 先求x，y的模 #|x|=sqrt(x1^2+x2^2+...+xn^2)
        # reduce_sum函数在指定维数上进行求和操作
        # （0-列，1-行）或者说，轴用来为超过一维的数组定义的属性，二维数据拥有两个轴：第0轴沿着行的垂直往下处理，第1轴沿着列的方向水平延伸处理
        x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))
        y_norm = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1))
        ### 求x和y的内积，即x*y。之所以用到tf.reduce_sum因为，内积公式为x1*y1+x2*y2+...+...
        x_y = tf.reduce_sum(tf.multiply(x, y), axis=1)
        ### 内积除以模的乘积，tf.multiply是模的乘积
        d = tf.divide(x_y, tf.multiply(x_norm, y_norm))
        return d


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    # shape[0]就是读取矩阵第一维度的长度
    # shape[1]就是读取矩阵第二维度的长度
    # sequence_length = x_train.shape[1],# x_train.shape[1]句子的个数，x_train.shape[0]样本的个数，句子长度
    # # 单个句子的最大长度（1个句子中单词的个数） max_document_length
    # num_classes = y_train.shape[1],  # 分类的种类0,1，在这就是2
    # vocab_size = len(vocab_processor.vocabulary_),  # 词汇表单词个数
    # embedding_size = FLAGS.embedding_dim,  # 向量化的维度128
    # filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),  # 卷积层的层数
    #  map（A，B）为映射，表示把A作为规则作用到B上
    # num_filters = FLAGS.num_filters,  # 每个卷积层的卷积核（过滤器）个数128
    # l2_reg_lambda: l2规则的参数λ

    def __init__(
            self, sequence_length, num_classes,index,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        # dropout 对于神经网络单元，按照一定的概率将其暂时从网络中丢弃,防止过拟合
        # input_x是输入占位，格式位[A，B] None表示该位不确定大小，sequence_length表示句长
        # input_y和上面的input_x基本应该是差不多的，格式位[A,B]的num_classes表示分类的类别数
        # input_y表示期望值？？
        # [None, sequence_length]应该是一个None * sequence_length的矩阵
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        # l2正则化损失的初始化为0
        l2_loss = tf.constant(0.0)

        # Embedding layer    # 该层构造的是输入向量维度格式
        # tf.device('/cpu:0')强制cpu执行，设置一个命名空间
        # 直接把wiki字典的字向量变为全数组的形式，下标就是索引
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            wiki_vector = []
            wiki_data_path = 'D:\\tensorflow\wiki.vector'
            wiki_dictionary = t_wiki_process.get_wiki_dic(wiki_data_path)
            count = 0    #暂时输出100个看效果
            for key in wiki_dictionary:
                #if count <= 100:
                word_embedding = wiki_dictionary[key]
                wiki_vector.append(np.array(word_embedding))
                    #count += 1
            self.embedded_word = (wiki_vector)
            #self.embedded_word = np.array(self.embedded_word)


            self.embedding = tf.Variable(tf.to_float(self.embedded_word), trainable=True, name="W")
            #print(self.embedded_word)
            self.embedded_chars = tf.nn.embedding_lookup(self.embedding, self.input_x)
            # expand_dims 增加维度（默认增加1维） #-1表示在最后一维,参考下面的输出来理解，增加一个维度适和内似于图像一样的结构
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            print("############################")
            print(self.embedded_chars)
            # Tensor("embedding/embedding_lookup:0", shape=(?, 56, 128), dtype=float32, device=/device:CPU:0)
            # 之所以为embedding/embedding_lookup:0,是因为采用了该方法（tf.nn.embedding_lookup），以该方法命名
            # 40 是单个句子最大长度，128是单个词的维度
            print(self.embedded_chars_expanded)
            # Tensor("embedding/ExpandDims:0", shape=(?, 56, 128, 1), dtype=float32, device=/device:CPU:0)
            print("############################")
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        # 3层 conv-maxpool-1 ,conv-maxpool-2,conv-maxpool-3+
        # 一层卷积得到h,一层max得到pooled，三个2层的网络的所有加和进 pooled_outputs
        # enumerate()，利用字典的形式遍历索引和元素，具体见http://blog.csdn.net/churximi/article/details/51648388
        # 字典里面应该是{0：3， 1：4， 2：5}  # 因为过滤器（卷积核）是None*3/4/5 三个，但是按照后面的输出结果，感觉应该是
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                print("filter_size, embedding_size, 1, num_filters", filter_size, embedding_size, 1, num_filters)
                # filter_size, embedding_size, 1, num_filters      3 128 1 128
                # 这就是卷积核的规格，3*128--高*宽，1为通道数（对应图像），128为卷积之后的下一层的深度
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # truncated_normal 从截断的正态分布中输出随机值  # 即卷积核的规格确定，但是卷积核的权重变量的数据随机生成
                # 卷积核下一层的深度是128，即上述的num_filters,
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # http://blog.csdn.net/mao_xiao_feng/article/details/78004522
                # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)  实现卷积功能的函数的参数规格
                # input
                # 指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]
                # 这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，
                # 注意这是一个4维的Tensor，要求类型为float32和float64其中之一
                # 实际：shape=(?, 40, 128, 1)
                # filter： 相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]
                # 这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，
                # 有一个地方需要注意，第三维in_channels，就是参数input的第四维
                # strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
                # string类型的量，只能是”SAME”,”VALID”其中之一，这个值决定了不同的卷积方式（后面会介绍）
                # 参考这篇文章去看 http://blog.csdn.net/mao_xiao_feng/article/details/78004522
                conv = tf.nn.conv2d(
                    input=self.embedded_chars_expanded,
                    filter=W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity  #采用非线性化激活
                # 计算修正线性单元(非常常用)：max(features, 0).并且返回和feature一样的形状的tensor
                # 参数：
                # features: tensor类型，必须是这些类型：A Tensor. float32, float64, int32, int64, uint8, int16, int8, uint16, half.
                # 加了偏置之后再激活
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                # http://blog.csdn.net/mao_xiao_feng/article/details/53453926
                # 第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature
                # map，依然是[batch, height, width, channels]  这样的shape
                # 第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
                # 第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride, stride, 1]
                # 第四个参数padding：和卷积类似，可以取'VALID' 或者 'SAME'
                # 返回一个Tensor，类型不变，shape仍然是 [batch, height, width, channels] 这种形式
                pooled = tf.nn.max_pool(
                    value=h,
                    # ksize为池化层的窗口规格，第一，四参数设为1，第二个参数为为卷积之后输出的窗口规格（高宽都是该公式），也就是池化的输入规格
                    # sequence_length 根据需求后面定义的,  filter_size分别为3，4，5
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        # filter_sizes 3,4,5  3层
        # num_filters 每层里面卷积核的个数
        # 128 * 3
        num_filters_total = num_filters * len(filter_sizes)
        print("pooled_outputs $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # 这是原本数据的输出 ,过滤器为3/4/5的时候
        # [ < tf.Tensor        # 'conv-maxpool-3/pool:0'        # shape = (?, 1, 1, 128) dtype = float32 >,
        # < tf.Tensor        # 'conv-maxpool-4/pool:0'        # shape = (?, 1, 1, 128) dtype = float32 >,
        #  < tf.Tensor        # 'conv-maxpool-5/pool:0'        # shape = (?, 1, 1, 128) dtype = float32 >]
        print("pooled_outputs   ", pooled_outputs)

        # Tensor("concat:0", shape=(?, 1, 1, 384), dtype=float32)
        # 在第三维上连接,在维度上shape = (?, 1, 1, 128)，分别为第0维，第1-3维
        self.h_pool = tf.concat(pooled_outputs, 3)
        print("h_pool ", self.h_pool)
        # 调整矩阵维度
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # Tensor("Reshape:0", shape=(?, 384), dtype=float32)
        print("h_pool_flat ", self.h_pool_flat)

        # Add dropout 防止过拟合
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            print("h_drop $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(self.h_drop)

        # Final (un-normalized 非标准的) scores and predictions
        # 这个num_classes就是最终输出的种类，这里是2分类
        # 这一层是全连接层？
        print(" Final (unnormalized) scores and predictions ")
        print("shape = [num_filters_total, num_classes]", [num_filters_total, num_classes])
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())  # 一种权值矩阵的初始化方式
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # 输出第一次损失，即初始值，为0
            print("l2_loss 1,", l2_loss)
            # 直接调用tf.nn.l2_loss函数算来l2正则损失
            # l2正则化对输入 W 和 b 起到作用，和交叉熵直接作用与输出不一样
            l2_loss += tf.nn.l2_loss(W)
            print("l2_loss 2,", l2_loss)
            l2_loss += tf.nn.l2_loss(b)
            print("l2_loss 3 ,", l2_loss)
            # tf.nn.xw_plus_b((x, weights) + biases.)
            #  相当于matmul(x, weights) + biases. 计算前向传播的结果
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            print("self.scores ,", self.scores)

            #############################
            #  这一步给出了预测的分类结果 #
            #############################
            # tf.argmax(input, axis=None, name=None, dimension=None) 此函数是对矩阵按行或列计算最大值
            # # axis：0表示按列，1表示按行
            # 这么理解，score是一个 self.scores , Tensor("output/scores:0", shape=(?, 2), dtype=float32)
            # 不知道多少行，但是是2列的矩阵
            # axis =1 ，就是取出每一行中，最大值所在的下标，0或1，也就是最终分类是0（第一类） 还是1（第二类）
            # 也就是正例还是反例
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            print("self.predictions ,", self.predictions)

        # Calculate mean cross-entropy loss  # 计算平均交叉熵损失
        with tf.name_scope("loss"):
            # 第一个参数logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，
            # 单样本的话，大小就是num_classes
            # 第二个参数labels：实际的标签，大小同上，为num_classes
            # 详细可见http://blog.csdn.net/mao_xiao_feng/article/details/53382790
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            # 求平均值
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy  # 利用预测得到的结果和期望得到的结果进行比较，计算精度
        with tf.name_scope("accuracy"):
            # tf.euqal输出true和false
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            # tf.cast类型转换函数，把数据(上述的True/False)映射变为float（浮点）类型
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            # softmax分类可参考这篇文章https://www.douban.com/group/topic/102669435/
            # 或者该文章最后一部分http://blog.csdn.net/cy_tec/article/details/52046806

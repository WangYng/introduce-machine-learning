import numpy
import tensorflow as tf
from matplotlib import pyplot as plt

print(tf.version.VERSION)

# 文档：https://www.tensorflow.org/tutorials/images/cnn?hl=zh-cn
# 训练一个卷积神经网络 (CNN) 模型来对图像进行分类
if __name__ == '__main__':
    # 加载数据集
    cifar10 = tf.keras.datasets.cifar10
    (train_images, train_nums), (test_images, test_nums) = cifar10.load_data()

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 数组类型 整数转小数
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    print("训练集大小 %d" % len(train_images))
    plt.figure(figsize=(10, 10))
    plt.title("train images")
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_nums[i][0]])
    plt.show()

    print("测试集大小 %d" % len(test_images))
    plt.figure(figsize=(10, 10))
    plt.title("test images")
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[test_nums[i][0]])
    plt.show()

    # 模型
    model = tf.keras.Sequential([
        # 输入的是 (32, 32, 3) 的图片, 经过卷积，输出的是形状为（30, 30, 32）的卷积基
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),

        # 输入的是形状为 (32, 32, 3) 的卷积基, 通过池化收缩宽度和高度，输出的是形状为（15, 15, 32）的卷积基
        tf.keras.layers.MaxPooling2D((2, 2)),

        # 输入的是形状为 (15, 15, 32) 的卷积基, 经过卷积，输出的是形状为（13, 13, 64）的卷积基
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

        # 输入的是形状为 (13, 13, 64) 的卷积基, 收缩宽度和高度，输出的是形状为（6, 6, 64）的卷积基
        tf.keras.layers.MaxPooling2D((2, 2)),

        # 输入的是形状为 (6, 6, 64) 的卷积基, 收缩宽度和高度，输出的是形状为（4, 4, 64）的卷积基
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

        # 将卷积基从三维数组（4, 4, 64）转换成一维数组
        tf.keras.layers.Flatten(),

        # 全连接神经层, 使用 relu 作为激活函数
        # 激活函数负责将神经元的输入映射到输出端,
        tf.keras.layers.Dense(64, activation='relu'),

        # 全连接神经层, 使用 softmax 作为激活函数。softmax 函数会返回一个概率值[0,1]，表示属于这个分类的可能性
        tf.keras.layers.Dense(10, activation='softmax'),

    ])

    # 评价指标(metrics) - 用于监控训练和测试步骤。以下示例使用了准确率，即被正确分类的图像的比率。
    # 损失函数(loss) - 测量模型在训练期间的准确程度。你希望最小化此函数，以便将模型“引导”到正确的方向上。
    # 优化算法(optimizer) - 决定模型如何根据其看到的数据和自身的损失函数进行更新。
    # 编译模型
    model.compile(metrics=['accuracy'], loss='sparse_categorical_crossentropy', optimizer='adam')

    model.summary()

    # 训练模型, 这样命名是因为该方法会将模型与训练数据进行“拟合”
    # 在模型训练期间，会显示损失和准确率指标
    model.fit(train_images, train_nums, epochs=10)

    # # 保存模型
    # model.save('./models/cnn_model.keras')
    #
    # # 加载模型
    # model = tf.keras.models.load_model('./models/cnn_model.keras')

    # 评估模型准确率
    model.evaluate(test_images, test_nums, verbose=2)

    # 使用模型，对输入的图片进行预测
    predict_images = test_images[25:50]
    predictions = model.predict(predict_images)
    numpy.set_printoptions(precision=2, suppress=True)

    plt.figure(figsize=(10, 10))
    plt.title("predict image")
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(predict_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[predictions[i].argmax()])
    plt.show()

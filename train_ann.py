import numpy
import tensorflow as tf
from matplotlib import pyplot as plt

print(tf.version.VERSION)

# 文档: https://www.tensorflow.org/tutorials/quickstart/beginner?hl=zh-cn
# 训练一个神经网络模型，可以识别手写数字

# 文档: https://www.tensorflow.org/tutorials/keras/classification?hl=zh-cn
# 训练一个神经网络模型，对运动鞋和衬衫等服装图像进行分类
if __name__ == '__main__':
    # 加载数据集
    mnist = tf.keras.datasets.mnist
    (train_images, train_nums), (test_images, test_nums) = mnist.load_data()

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # 加载数据集
    # fashion_mnist = tf.keras.datasets.fashion_mnist
    # (train_images, train_nums), (test_images, test_nums) = fashion_mnist.load_data()
    #
    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

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
        plt.xlabel(class_names[train_nums[i]])
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
        plt.xlabel(class_names[test_nums[i]])
    plt.show()

    # 模型
    model = tf.keras.models.Sequential([
        # 将图像格式从二维数组（28 x 28 像素）转换成一维数组
        tf.keras.layers.Flatten(input_shape=(28, 28)),

        # 全连接神经层, 使用 ReLU 作为激活函数
        tf.keras.layers.Dense(8, activation='relu'),

        # 全连接神经层, 使用 softmax 作为激活函数。softmax 函数会返回一个概率值[0,1]，表示属于这个分类的可能性
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    print(model.summary())

    # 评价指标(metrics) - 用于监控训练和测试步骤。以下示例使用了准确率，即被正确分类的图像的比率。
    # 损失函数(loss) - 测量模型在训练期间的准确程度。你希望最小化此函数，以便将模型“引导”到正确的方向上。
    # 优化算法(optimizer) - 决定模型如何根据其看到的数据和自身的损失函数进行更新。
    # 编译模型
    model.compile(metrics=['accuracy'], loss='sparse_categorical_crossentropy', optimizer='adam')

    # 训练模型, 这样命名是因为该方法会将模型与训练数据进行“拟合”
    # 在模型训练期间，会显示损失和准确率指标
    model.fit(train_images, train_nums, epochs=10)

    # # 保存模型
    # model.save('./models/ann_model.keras')
    #
    # # 加载模型
    # model = tf.keras.models.load_model('./models/ann_model.keras')

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

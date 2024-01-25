import time

import tensorflow as tf

print(tf.version.VERSION)


class MyRNNModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)

        # 将每个词表示为 EMBEDDING_SIZE 维，其实意味着这 EMBEDDING_SIZE 维中的每个维，都单独表示了词的每个属性，
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # 门控循环单元
        self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)

        # 全连接神经层
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')] * len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        # Apply the prediction mask: prevent "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states


def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


# 文档：https://www.tensorflow.org/text/tutorials/text_generation
# 训练一个循环神经网络 (RNN) 模型，来预测序列中的下一个字符（“e”）。通过重复调用模型可以生成更长的文本序列。
if __name__ == '__main__':
    path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                           'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

    # 文本内容
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

    # 将文本向量转换为字符流
    vocab = sorted(set(text))
    ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
    chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True,
                                                  mask_token=None)

    all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
    ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

    # 将这些单独的字符转换为所需大小的序列
    sequences = ids_dataset.batch(101, drop_remainder=True)

    # 为了进行训练，您需要一个(input, label)成对的数据集，并对数据进行乱序
    dataset = sequences.map(split_input_target)
    for input_example, target_example in dataset.take(1):
        print("Input :", text_from_ids(input_example).numpy())
        print("Target:", text_from_ids(target_example).numpy())
    dataset = (dataset.shuffle(10000).batch(64, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE))

    # Length of the vocabulary in StringLookup Layer
    vocab_size = len(ids_from_chars.get_vocabulary())

    # The embedding dimension
    embedding_dim = 256

    # Number of RNN units
    rnn_units = 1024

    model = MyRNNModel(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units)

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)

    # 损失函数(loss) - 测量模型在训练期间的准确程度。你希望最小化此函数，以便将模型“引导”到正确的方向上。
    # 优化算法(optimizer) - 决定模型如何根据其看到的数据和自身的损失函数进行更新。
    # 编译模型
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    model.summary()

    # 训练模型, 这样命名是因为该方法会将模型与训练数据进行“拟合”
    # 在模型训练期间，会显示损失和准确率指标
    model.fit(dataset, epochs=20)

    # 保存模型
    model.save('./models/my_rnn_model.keras')

    # # 加载模型
    # model = tf.keras.models.load_model('./models/my_rnn_model.keras')

    # 生成文本, 使用此模型生成文本的最简单方法是循环运行它
    one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

    start = time.time()
    states = None
    next_char = tf.constant(['ROMEO:'])
    result = [next_char]

    for n in range(1000):
        next_char, states = one_step_model.generate_one_step(next_char, states=states)
        result.append(next_char)

    result = tf.strings.join(result)
    end = time.time()
    print(result[0].numpy().decode('utf-8'), '\n\n' + '_' * 80)
    print('\nRun time:', end - start)

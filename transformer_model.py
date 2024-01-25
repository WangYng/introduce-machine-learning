import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def translation(model_checkpoint, sentence):
    # 创建模型
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    model = model.to(device)

    # 生成 token ID 序列
    sentence_input = tokenizer(sentence, return_tensors="pt").to(device)
    sentence_generated_tokens = model.generate(
        sentence_input["input_ids"],
        attention_mask=sentence_input["attention_mask"],
        max_length=128
    )

    # token ID 序列 转换为文本
    return tokenizer.decode(sentence_generated_tokens[0], skip_special_tokens=True)


# Transformer 是一种深度神经网络，用自注意力机制取代了 CNN 和 RNN 。自注意力使 Transformer 能够轻松地跨输入序列传输信息。
# tensowflow 文档：https://www.tensorflow.org/text/tutorials/transformer
# Transformers 是由 Hugging Face 开发的一个 NLP 包，使用 Transformer 模型编写，支持加载目前绝大部分的预训练模型。
# 文档：https://transformers.run
if __name__ == '__main__':

    # 英文转中文
    sentence = 'Hugging Face is an American company that develops tools for building machine learning applications.'
    predict = translation("Helsinki-NLP/opus-mt-en-zh", sentence)
    print("英文转中文\n原文：%s\n译文：%s\n" % (sentence, predict))

    # 中文转英文
    sentence = 'Hugging Face是一家美国公司，专门开发用于构建机器学习应用的工具。'
    predict = translation("Helsinki-NLP/opus-mt-zh-en", sentence)
    print("中文转英文\n原文：%s\n译文：%s\n" % (sentence, predict))


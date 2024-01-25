from transformers import AutoModelForCausalLM, AutoTokenizer

# 大语言模型
# 模型来自与Hugging Face。 号称强过ChatGPT3.5。运行需要28G内存。如果没有3080Ti，推理一次结果大约需要10分钟。
# https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
if __name__ == '__main__':
    checkpoint = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)  # You may want to use bfloat16 and/or move to GPU here

    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "Use java code write a url extractor tool",
        },
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_tensors="pt")

    outputs = model.generate(tokenized_chat, max_new_tokens=512, eos_token_id=2, pad_token_id=2)
    print(tokenizer.decode(outputs[0]))

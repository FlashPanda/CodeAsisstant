from transformers import AutoTokenizer
import pickle

tokenizerGpt2 = AutoTokenizer.from_pretrained("gpt2")
input_text = "Write a Python function to calculate factorial"
tokenized_input = tokenizerGpt2(input_text, return_tensors="tf")
print(tokenized_input)

tokenizerBert = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenized_input = tokenizerBert(input_text, return_tensors="tf")
print(tokenized_input)


with open(r'..\Data\python_licenses.pkl', 'rb') as f:
    data = pickle.load(f)
# 如果 data 是一个列表，显示前 5 个元素
if isinstance(data, list):
    print('this is a list.')
    print(data[:5])

# 如果 data 是一个字典，显示前 5 个键值对
elif isinstance(data, dict):
    print('this is a dict')
    keys = list(data.keys())[:5]  # 取前 5 个键
    # for key in keys:
    #     print(f"{key}: {data[key]}")

count = 1
for tempKey, tempValue in data:
    count += 1
    if count > 5:
        break
    print(f'tempKey:{tempKey} : tempValue:{tempValue}')

# # 提取文本内容
# texts = []
# for repo_name, files in data:
#     for file_path, file_content in files:
#         texts.append(file_content.strip())  # 去除首尾空白字符

# for text in texts:
#     encoded_dict = tokenizer.encode_plus(
#         text,
#         add_special_tokens=True,   # 添加特殊标记，如 <|endoftext|>
#         max_length=1024,           # GPT-2 的最大长度为 1024
#         truncation=True,           # 超过最大长度则截断
#         padding='max_length',      # 填充到最大长度
#         return_attention_mask=True,
#         return_tensors='tf',       # 返回 TensorFlow tensors
#     )
#     input_ids.append(encoded_dict['input_ids'])
#     attention_masks.append(encoded_dict['attention_mask'])

# # 将列表转换为张量
# input_ids = tf.concat(input_ids, axis=0)
# attention_masks = tf.concat(attention_masks, axis=0)

# # 创建数据集
# dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks))

# # 定义函数创建输入和标签
# def map_func(input_id, attention_mask):
#     labels = tf.concat([input_id[:, 1:], tf.constant([[tokenizer.eos_token_id]], dtype=tf.int32)], axis=1)
#     return {'input_ids': input_id, 'attention_mask': attention_mask}, labels

# dataset = dataset.map(map_func)

# # 设置批量大小和混洗缓冲区大小
# batch_size = 4
# dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)

# # 加载预训练的 GPT-2 模型
# model = TFGPT2LMHeadModel.from_pretrained('gpt2')

# # 编译模型
# optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# model.compile(optimizer=optimizer, loss=loss)

# # 训练模型
# epochs = 3
# model.fit(dataset, epochs=epochs)
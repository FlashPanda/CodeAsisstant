from transformers import AutoTokenizer
import pickle
import tensorflow as tf
from transformers import TFGPT2LMHeadModel
from transformers import AdamW

tokenizerGpt2 = AutoTokenizer.from_pretrained("gpt2")
tokenizerGpt2.pad_token = tokenizerGpt2.eos_token
input_text = "Write a Python function to calculate factorial"
#tokenized_input = tokenizerGpt2(input_text, return_tensors="tf")
#print(tokenized_input)

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
    keys = list(data.keys())[:10]  # 取前 5 个键
    # for key in keys:
    #     print(f"{key}: {data[key]}")

inputs_ids = []
outputs_ids = []
input_attention_masks = []
output_attention_masks = []
for oneKey in keys:
    #print(f'item0{data[oneKey][0]}')
    input_text, output_text = data[oneKey][0]
    # print(f'input_text: {input_text}')
    # print(f'output_text: {output_text}')
    input_encoding = tokenizerGpt2.encode_plus(
        input_text,
        add_special_tokens=True,    # 添加特殊标记 [CLS], [SEP]
        max_length=1024,              # 最大长度（会自动截断）
        padding='max_length',       # 填充到最大长度
        truncation=True,            # 自动截断过长的文本
        return_attention_mask=True, # 返回 attention mask
        return_tensors='tf'         # 返回 PyTorch tensors 格式
    )
    output_encoding = tokenizerGpt2.encode_plus(
        output_text, 
        add_special_tokens=True, 
        max_length=1024, 
        padding='max_length', 
        truncation=True,            # 自动截断过长的文本
        return_attention_mask=True,
        return_tensors='tf')
    print(f'input_encoding: {input_encoding}')
    print(f'output_encoding: {output_encoding}')
    inputs_ids.append(input_encoding['input_ids'])
    input_attention_masks.append(input_encoding['attention_mask'])
    outputs_ids.append(output_encoding['input_ids'])
    output_attention_masks.append(output_encoding['attention_mask'])

dataset = tf.data.Dataset.from_tensor_slices((
    {
        'input_ids': tf.concat(inputs_ids, axis=0),
        'attention_mask': tf.concat(input_attention_masks, axis=0),
    },
    {
        'output_ids': tf.concat(outputs_ids, axis=0),
        'output_attention_mask': tf.concat(output_attention_masks, axis=0),
    }
))

# 按批次加载
batch_size = 16
dataset = dataset.batch(batch_size)

model = TFGPT2LMHeadModel.from_pretrained('gpt2')

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
#optimizer = AdamW(model.trainable_variables, lr=5e-5)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer=optimizer, loss=loss_fn)

epochs = 3  # 训练轮数
model.fit(dataset, epochs=epochs)
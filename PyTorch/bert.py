# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# 初始化分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 加载预训练的BERT模型
model = BertModel.from_pretrained('bert-base-uncased')

# 示例文本
text = "Here is some text to encode"

# 将文本转换为token
encoded_input = tokenizer(text, return_tensors='pt')
# 将输入传递给模型
output = model(**encoded_input)

# 输出的隐藏状态
hidden_states = output.last_hidden_state
# 初始化一个MLM模型
mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# 假设我们要预测被遮蔽的单词
masked_text = "Here is some [MASK] to encode"
masked_input = tokenizer(masked_text, return_tensors='pt')

# 使用MLM模型预测被遮蔽的单词
mlm_output = mlm_model(**masked_input)

# 获取预测结果
predictions = mlm_output.logits
# 查找被遮蔽位置的索引
mask_index = torch.where(masked_input['input_ids'][0] == tokenizer.mask_token_id)[0]

# 获取预测结果中最可能的单词
predicted_token_id = predictions[0, mask_index].argmax(dim=1)
predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)[0]

print("Predicted token:", predicted_token)

练习 2.3 词元嵌入与位置嵌入

- 新增 token_embedding_layer（50257×256）与 pos_embedding_layer（4×256）
- 将两者相加得到 input_embeddings，输出形状 [8, 4, 256]
- 设置 torch.manual_seed(123) 保证结果可复现
- 删除旧的 batch_size=1 演示代码
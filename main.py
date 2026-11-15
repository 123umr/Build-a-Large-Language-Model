import tiktoken 
import torch 
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset): 
    def __init__(self, txt, tokenizer, max_length, stride): #一个用于批处理输入和目标的数据集
        self.input_ids = [] 
        self.target_ids = [] 

        token_ids = tokenizer.encode(txt) # 对文本进行分词

        for i in range(0, len(token_ids) - max_length, stride): 
            input_chunk = token_ids[i:i + max_length] 
            target_chunk = token_ids[i + 1: i + max_length + 1] 
            self.input_ids.append(torch.tensor(input_chunk)) 
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self): # 返回数据集的大小
        return len(self.input_ids)

    def __getitem__(self, idx): # 返回数据集的第 idx 个样本
        return self.input_ids[idx], self.target_ids[idx] 

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")   # 初始化分词器
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)   # 创建数据集
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader 

with open("the-verdict.txt", "r", encoding="utf-8") as f: 
    raw_text = f.read()


max_length = 4
dataloader = create_dataloader_v1(
 raw_text, batch_size=8, max_length=max_length,
 stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
# 输入批数据 目标批数据
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# 利用 vocab_size 和 output_dim 在 PyTorch 中实例化一个嵌入层。
vocab_size = 50257
output_dim = 256
torch.manual_seed(123)# 为了确保结果的可重复性，可以将随机种子设置为 123
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

#  相同维度的嵌入层  为了获取 GPT 模型所采用的绝对位置嵌入
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
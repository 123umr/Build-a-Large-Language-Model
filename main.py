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

dataloader = create_dataloader_v1( 
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False) 
data_iter = iter(dataloader) 
first_batch = next(data_iter) 
print(first_batch)

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4,
    shuffle=False
    )
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets) 
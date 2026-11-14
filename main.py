from importlib.metadata import version 
import tiktoken 
print("tiktoken version:", version("tiktoken")) 

tokenizer = tiktoken.get_encoding("gpt2")

text = ( 
"我丢了一部苹果手机" 
) 
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"}) 
print(integers)

for token_id in integers:
    print(token_id, "->", repr(tokenizer.decode([token_id])))

strings = tokenizer.decode(integers) 
print(strings)
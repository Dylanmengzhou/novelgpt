
import load_data as ld
import model as md
import torch

torch.manual_seed(1337)
# read the text file
text = ld.load_data('input.txt')


# preprocess the text
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
block_size = 256
model = md.BigramLanguageModel(vocab_size)
model.load_state_dict(torch.load('model.pth'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
model.eval()  # 设置为评估模式，这通常会关闭dropout等

# 3. 准备prompt
prompt_text = "谣言"
prompt_tokens = encode(prompt_text)  # 使用之前定义的encode函数将文本转换为tokens
prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

# 4. 生成文本
generated_tokens_tensor = model.generate(prompt_tensor, max_new_tokens=500)
generated_text = decode(generated_tokens_tensor[0].tolist())  # 使用之前定义的decode函数

# 打印生成的文本
print(generated_text)


#download the model from modelscope and saved in model folder


from modelscope import snapshot_download

model_dir = snapshot_download(
    'LLM-Research/Meta-Llama-3-8B',
    cache_dir='/home/zceexl3/ai_accountant/models',
    allow_patterns=['*.safetensors', 'tokenizer.*', '*.json', '*.md'],
    ignore_patterns=['original/*']  # ignore consolidated.00.pth, no need 
)





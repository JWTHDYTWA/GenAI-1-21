import torch
from transformers import pipeline

models = {
    "q3": "Qwen/Qwen3-4B-Instruct-2507",
    "g2": "google/gemma-2-2b"
}

print('Initializing a pipeline...')
pipe = pipeline(
    task="text-generation",
    model=models["q3"],
    device_map="auto",
    dtype="auto",
)
print('A pipeline has been initialized.')


style = 'деловом'
while True:
    text = input('Введите фразу: ')
    if text == ':q':
        break
    else:
        message = [
            {'role': 'system', 'content': f'Твоя задача — перефразировать текст, который присылает пользователь, в {style} стиле. Не добавляй ничего от себя, даже кавычки.'},
            {'role': 'user', 'content': f'Текст: "{text}"'}
        ]
        answer = pipe(message)[0]['generated_text'][-1]['content']
        print(answer)
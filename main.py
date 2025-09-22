import argparse
from transformers import pipeline, TextGenerationPipeline


def text_pipeline_init(lm:str, padding:str):
    """
    Создаёт экземпляр пайплайна для генерации текста с заданной моделью и настройками.

    Args:
        lm (str): Имя или путь к языковой модели.
        padding (str): Тип выравнивания токенов ('left' или 'right').

    Returns:
        TextGenerationPipeline: Инициализированный пайплайн для генерации текста.
    """
    pipe = pipeline(
        task="text-generation",
        model=lm,
        # Для автоматического подбора следующих параметров используется библиотека `accelerate`
        device_map="auto",
        dtype="auto"
    )
    # Левый padding нужен чтобы модель не теряла контекст при обработке списка
    pipe.tokenizer.padding_side = padding
    return pipe


def inference(style:str, input:str|list[str], pipe: TextGenerationPipeline, token_limit=None):
    """
    Перефразирует входной текст в указанном стиле с использованием заданного пайплайна.

    Args:
        style (str): Стиль, в котором необходимо перефразировать текст.
        input (str | list[str]): Строка или список строк для обработки.
        pipe (TextGenerationPipeline): Инициализированный пайплайн TextGenerationPipeline из библиотеки transformers.

    Returns:
        list[str]: Список перефразированных текстов.

    Raises:
        TypeError: Если параметр `input` не является строкой или списком строк.
    """

    message = None
    answer = None

    if not isinstance(style, str):
        raise TypeError('Аргумент style должен иметь тип str.')
    if not isinstance(pipe, TextGenerationPipeline):
        raise TypeError('Аргумент pipe должен иметь тип TextGenerationPipeline.')

    # Разные алгоритмы для обработки одиночной строки и списка строк

    if isinstance(input, str):
        if input.strip() == '':
            raise ValueError('Входная строка не должна быть пустой или состоять только из пробелов.')
        message = [
            {'role': 'system', 'content': f'Твоя задача — перефразировать в указанном стиле текст, который присылает пользователь. Не добавляй ничего от себя, даже кавычки.'},
            {'role': 'user', 'content': f'Текст: "{input}"\nСтиль: {style}'}
        ]
        answer = pipe(message, max_new_tokens=token_limit)[0]['generated_text'][-1]['content']

    elif isinstance(input, list):
        message = []
        for ln in input:
            if not isinstance(ln, str):
                continue
            elif ln.strip() == '':
                # К сожалению, сложно оставить разделяющие строки
                # чтобы сохранить структуру входного файла, поэтому
                # они просто пропускаются.
                continue
            else:
                message.append(
                    [
                        {'role': 'system', 'content': f'Твоя задача — перефразировать в указанном стиле текст, который присылает пользователь. Не добавляй ничего от себя, даже кавычки.'},
                        {'role': 'user', 'content': f'Текст: "{ln}"\nСтиль: {style}'}
                    ]
                )
        if not message:
            raise ValueError('Список запросов не должен быть пустым.')
        answer = pipe(message, batch_size=16, max_new_tokens=token_limit)
        answer = [ a[0]['generated_text'][-1]['content'] for a in answer ]

    else:
        raise TypeError('Аргумент input должен принимать строку или список строк.')
    
    return answer

# Привязка к конкретной модели вызвана отличающимся форматом данных у разных моделей (проверено).
# При переключении модели может сломаться индексация контейнеров, из-за чего потребуется переписывать код.
# Примечание: Qwen3, в отличии от Gemma, не является Gated моделью и не требует токена HuggingFace
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_STYLE = "Официальный"
MAX_TOKENS = 50

def main():

    # Используется библиотека argparse для обработки параметров командной строки
    # Описание параметров командной строки можно получить, вызвав программу с флагом -h:
    # ./main.py -h
    parser = argparse.ArgumentParser('GenAI-1-21')
    parser.add_argument('input_file', nargs='?', default='input.txt', help='Путь к входному файлу. По умолчанию - "input.txt".')
    parser.add_argument('-o', '--output', default='output.txt', help='Путь к выходному файлу. По умолчанию - "output.txt".')
    parser.add_argument('-r', '--realtime', action='store_true', help='Запуск в режиме реального времени.')
    parser.add_argument('-s', '--style', default=DEFAULT_STYLE, help=f'Выбор стиля, в котором будет переписан текст. По умолчанию - "{DEFAULT_STYLE}".')
    parser.add_argument('-t', '--tokens', default=MAX_TOKENS, type=int, help=f'Лимит output-токенов на один запрос. Влияет на длину ответов. По умолчанию - {MAX_TOKENS}.')
    args = parser.parse_args()

    # Инициализация пайплайна модели

    try:
        pipe_instance = text_pipeline_init(MODEL_NAME, padding='left')
        print('Модель успешно инициализирована.')
    except Exception as e:
        print(f'\033[31mПроизошла ошибка при инициализации модели:\n{e}\033[0m')
        exit(1)

    
    if args.realtime:
        # Режим реального времени (диалог)
        print('Для выхода из программы введите \033[34m:q\033[0m.')
        while True:
            text = input('Введите фразу: ')
            if text == ':q':
                break
            elif text.strip() == '':
                print('Введена пустая строка.')
                continue
            else:
                try:
                    answer = inference(args.style, text, pipe_instance, args.tokens)
                    print(answer + '\n')
                except Exception as e:
                    print(f'Ошибка обработки текста:\n{e}')
    else:
        # Режим обработки файла
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                lines = f.read().splitlines()

            # Передача входных данных модели и получение ответа
            answer = inference(args.style, lines, pipe_instance, args.tokens)

            with open(args.output, 'w', encoding='utf-8') as f:
                f.write('\n'.join(answer))
            
        except FileNotFoundError as e:
            print(f'\033[31mОшибка открытия файла:\n{e}\033[0m')
        except Exception as e:
            print(f'\033[31mОшибка обработки файла:\n{e}\033[0m')


if __name__ == '__main__':
    main()

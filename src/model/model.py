from transformers import MBartTokenizer, MBartForConditionalGeneration

model_name = "IlyaGusev/mbart_ru_sum_gazeta"


def load_tokenizer():
    """
    Returns
    -------
    tokenizer : MBartTokenizer
        Токенизатор
    """
    tokenizer = MBartTokenizer.from_pretrained(model_name)
    return (tokenizer)


def load_bart_model():
    """Загрузка предобученной модели для суммаризации текста.
    Модель обучена на массиве статей с сайта gazeta.ru.
    Модель может неадекватно обрабатывать некоторые виды входящего текста.
    Returns
    -------
    model : MBartForConditionalGeneration
        Модель
    """
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    return (model)


def make_summary_text(tokenizer, model, text):
    """Функция суммаризации текста.

    Parameters
    ----------
    tokenizer : MBartTokenizer
        Токенизатор
    model : MBartForConditionalGeneration
        Модель
    text : str
        Текст для обработки моделью.

    Returns
    -------
     : str
        Суммаризированный текст.
    """

    if text == "":
        return "Введите текст в окно выше"  # Проверка на пустую строку

    model_inputs = tokenizer(
        [text],  # Тест для токенизации (разделения на части) и векторизации
        max_length=1024,  # Максимум модели 1024
        padding=False,  # токенов  меньше чем max_length, дополнить пустыми
        truncation=True,  # токенов больше чем max_length, включить усечение
        return_tensors="pt",  # библиотека PyTorch для векторизации
    )

    input_ids = model_inputs["input_ids"]  # Вектор входящего текста
    input_length = input_ids.shape[1]  # фактическая длина вектора
    _min_length = int(
        input_length * 0.1
    )  # Минимальное количество токенов установлено как 10% от input_length

    # Должно быть меньше или равно 200.
    min_length = (_min_length if _min_length <= 200 else 200)

    model_outputs = model.generate(
        input_ids=input_ids,  # Векторизированный текст передаётся модели
        min_length=min_length,
        no_repeat_ngram_size=4,  # Последовательности могут встречаться 1 раз
    )
    output_ids = model_outputs[0]  # Вектор исходящего текcта.

    # Декодирование - превращение вектора в текст.
    summary = tokenizer.decode(output_ids, skip_special_tokens=True)
    return summary

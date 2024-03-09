import torch
import streamlit as st
from src.model.model import load_tokenizer, load_bart_model,  make_summary_text


def launch_app():
    # определяем на чем будем запускать модель
    if torch.cuda.is_available():
        torch.device("cuda")
    else:
        torch.device("cpu")

    st.title("Краткий пересказ текстов")
    # получаем текст от клиента
    text = st.text_area("Введите текст")

    # загружаем модель и токенизатор. Кешируем в Streamlit
    @st.cache_resource()
    def load_model():
        tokenizer = load_tokenizer()
        model = load_bart_model()
        return (tokenizer, model)

    tokenizer, model = load_model()

    st.write("Краткий пересказ")

    if st.button("Применить"):
        # вывод решения на экран
        st.success(make_summary_text(tokenizer, model, text))

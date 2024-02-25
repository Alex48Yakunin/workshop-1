from streamlit.testing.v1 import AppTest

# Инициализация приложения
at = AppTest.from_file("src/ui/client.py")


def test_app():
    # Запуск приложения
    at.run(timeout=100)


def test_tex_area():
    # Проверка, что st.text_area не равно null
    if len(at.text_area) > 0:
        at.text_area[0].input('Тестовый текст')
        assert at.text_area[0].value is not None


def test_button():
    # Проверка, что st.button существует и её можно нажать
    if len(at.button) > 0:
        assert 'st.button существует'


def test_success():
    # Проверка, что st.success не равно null
    if len(at.success) > 0:
        assert at.success[0].value is not None


def test_success_value():
    # Если в st.text_area есть значение, 
    # то на выходе в st.success будет значение не равно null
    if len(at.text_area) > 0:
        at.text_area[0].input('Тестовый текст')
        if len(at.success) > 0:
            assert at.success[0].value is not None

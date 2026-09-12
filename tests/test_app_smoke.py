from streamlit.testing.v1 import AppTest


def test_app_page_renders_without_errors():
    app = AppTest.from_file("../streamlit_app.py", default_timeout=120).run()
    assert not app.exception
    assert app.title[0].value == "🔍 RAG Pipeline Explorer"

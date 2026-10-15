from streamlit.testing.v1 import AppTest


def test_app_page_renders_without_errors():
    app = AppTest.from_file("../streamlit_app.py", default_timeout=120).run()
    assert not app.exception
    # The page title lives in the hero block (st.html), not in st.title.
    assert any("RAG Pipeline Explorer" in element.proto.body for element in app.get("html"))

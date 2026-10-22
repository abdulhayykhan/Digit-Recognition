from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=120).run()
assert not at.exception, at.exception
print("loaded; tabs:", len(at.tabs))
# train a small model through the UI
at.button[0].click().run()
assert not at.exception, at.exception
print("trained; metrics:", [m.value for m in at.metric][:6])

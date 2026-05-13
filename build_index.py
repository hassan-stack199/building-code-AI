"""Pre-build the regulation index at Docker build time.

Run once during `docker build`. Reads all PDFs in ./regulations/, embeds them,
and saves the result to ./cache/. The cached file gets baked into the image,
so when the container starts the index is already there — no first-load wait.
"""

from __future__ import annotations

import os
import sys

# Streamlit isn't available at build time and we don't need it here.
# Stub out the streamlit decorators so importing app.py works.
import types

stub_st = types.ModuleType("streamlit")


def _passthrough(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    def decorator(fn):
        return fn
    return decorator


stub_st.cache_resource = _passthrough
stub_st.cache_data = _passthrough


class _Progress:
    def progress(self, *a, **kw):
        pass

    def empty(self):
        pass


stub_st.progress = lambda *a, **kw: _Progress()
stub_st.warning = print
stub_st.error = print
stub_st.info = print
stub_st.session_state = {}

sys.modules["streamlit"] = stub_st

# Now import the app and trigger the index build.
import app  # noqa: E402

print("[build_index] Building shared library index…")
idx = app.load_shared_index()
print(f"[build_index] Done. {len(idx.chunks)} chunks indexed.")

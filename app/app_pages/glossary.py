"""Glossary page: the project's methodology and architecture docs, in two tabs.

Hidden from the top navigation; opened from the user bar or at /glossary. The
docs are read from docs/ with their GitHub math fences converted for Streamlit.
"""

import shared
import streamlit as st

from src.simulator.docs import DocReader

reader = DocReader()
shared.explain(
    "How every number in this app is produced (methodology) and how the code is organised "
    "(architecture). The same documents live in the repository's docs/ folder."
)
methodology, architecture = st.tabs(
    [":material/functions: Methodology", ":material/account_tree: Architecture"]
)
with methodology:
    st.markdown(reader.read("methodology"))
with architecture:
    st.markdown(reader.read("architecture"))

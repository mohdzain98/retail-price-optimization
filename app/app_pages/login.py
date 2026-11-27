"""Sign-in page: centred two-column layout, project story left, sign-in card right.

The story and card header are static HTML (app/landing.py); the form is native.
Accounts are local (SQLite, salted scrypt hashes) and created with `make user`;
a successful sign-in stores the user in the session and reruns.
"""

import shared
import shell
import streamlit as st
from landing import LandingArt, Palette

store = shared.store()
art = LandingArt(Palette.for_theme(st.context.theme.type))

LANDING_KEY = "landing"
# Centre the page vertically: the keyed container is at least the window height
# (minus the top bar and page padding) and centres its content.
st.html(
    f"<style>.st-key-{LANDING_KEY} {{ min-height: calc(100vh - 7rem); justify-content: center; }}"
    "</style>"
)
with st.container(key=LANDING_KEY):
    _, centre, _ = st.columns([1, 10, 1])
    with centre:
        story, card = st.columns([7, 5], gap="xlarge", vertical_alignment="center")

with story:
    st.html(art.story())
    if st.button("About the tool", icon=":material/info:", type="tertiary", key="about_tool"):
        shell.about_dialog()

with card, st.container(border=True, gap="medium"):
    st.html(art.card_header())
    st.caption("Use the local account you created with `make user`.")
    if store.count() == 0:
        st.info(
            "No account on this machine yet. In a terminal run `make user`, then sign in here.",
            icon=":material/person_add:",
        )
    with st.form("sign_in", border=False):
        email = st.text_input(
            "Email",
            type="email",
            placeholder="name@gmail.com",
            autocomplete="email",
            key="login_email",
        )
        password = st.text_input(
            "Password",
            type="password",
            icon=":material/lock:",
            autocomplete="current-password",
            key="login_password",
        )
        submitted = st.form_submit_button(
            "Sign in", icon=":material/arrow_forward:", type="primary", width="stretch"
        )
    if submitted:
        account = store.authenticate(email, password) if email and password else None
        if account is None:
            st.error("Email or password is incorrect.", icon=":material/error:")
        else:
            shared.sign_in(account)
            st.rerun()
    st.caption(
        ":material/shield: Stored on this machine only. Passwords are kept as salted hashes."
    )

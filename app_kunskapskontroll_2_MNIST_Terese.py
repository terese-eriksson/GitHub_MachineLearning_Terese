import streamlit as st

pages = {
    "Home": [
        st.Page("menu_home1.py", title="About"),
    ],
    "Predict numbers": [
        st.Page("menu_predict_numbers1.py", title="Draw"),
        st.Page("menu_predict_numbers2.py", title="Upload image"),
    ],
}

pg = st.navigation(pages)
pg.run()
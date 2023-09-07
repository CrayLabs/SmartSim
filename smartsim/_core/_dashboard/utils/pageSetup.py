import streamlit as st
from PIL import Image

## Page Setup ##


# Add css
def local_css(file_name: str) -> None:
    with open(file_name, encoding="utf-8") as file:
        st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)


# Loading Image using PIL
im: Image.Image = Image.open("smartsim/_core/_dashboard/assets/SmartSim.png")


# Set page config
def set_streamlit_page_config() -> None:
    st.set_page_config(layout="wide", page_title="Dashboard", page_icon=im)

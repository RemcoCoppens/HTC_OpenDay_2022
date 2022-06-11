import streamlit as st
import s3fs
from PIL import Image
import io
# Retrieve file contents.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
@st.experimental_memo(ttl=600)
def read_image(filename):
    with fs.open(filename, mode='rb') as f:
        return f.read()

def bytes_to_image(bytes):
    return Image.open(io.BytesIO(bytes))

fs = s3fs.S3FileSystem(anon=False)


st.markdown("# Gallery 📸")
st.sidebar.markdown("# Gallery 📸")
col1, col2, col3, col4 = st.columns(4)

files = fs.ls('openday2022streamlit/Fotos_HTCopenday2022')[5:17]
bytes = [read_image(file) for file in files if file.endswith('.jpeg')]
images = [bytes_to_image(img) for img in bytes if len(img) > 0]

with col1:
    for i, image in enumerate(images):
        if i % 3 == 0:
            st.image(image, width=250)

with col2:
    for i, image in enumerate(images):
        if i % 3 == 1:
            st.image(image, width=250)

with col3:
    for i, image in enumerate(images):
        if i % 3 == 2:
            st.image(image, width=250)




# for fn in files:
#     if fn.endswith('.jpeg'):
        
#         # image = 
#         st.write("bytes_to_image(file)")
        
    


# with open(fn, "rb") as art:
#     btn = st.download_button(
#         label="Download artwork",
#         data=art,
#         file_name=fn,
#         mime="image/png"
#     )
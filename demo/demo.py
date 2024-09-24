import streamlit as st

from PIL import Image
from matplotlib import pyplot as plt

from src.facial_key_points.utils.facekeypoints import FacialKeyPointDetection

st.set_page_config(page_title="Facial Keypoint Detection", page_icon="F")

st.title("Facial Keypoint Detection")
st.markdown('Upload Face to Detect Facial Key Point ')

image = st.file_uploader('Upload Facial Image', ['jpg', 'jpeg', 'png'], accept_multiple_files=False )
facial_key_point_detection = FacialKeyPointDetection()

if image is not None:
    image=Image.open(image).convert('RGB')
    st.image(image)
    
    image, kp = facial_key_point_detection.predict(image)


    fig = plt.figure()
    plt.imshow(image)
    plt.scatter(kp[0], kp[1], s=4, c='r')
    st.pyplot(fig)


import streamlit as st
import joblib
import cv2
import numpy as np
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt

# Upload number
st.title("Upload image")
st.write("Make sure nothing but the number is showing in the picture.")

def center_resize_digit(img):
    # Find all non-black pixels = the number, and make rectangle around number. 
    coords = cv2.findNonZero(img)
    if coords is None:
        return np.zeros((28, 28), dtype=np.uint8)  # Return empty picture if nothing is found
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y+h, x:x+w]

    # Resize picture of number to 20 x 20 pixels 
    scale_factor = 20.0 / max(w, h)
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    resized = cv2.resize(cropped, (new_w, new_h))  # Ändra storlek

    # Center picture of number on a 28 x 28 pixel black canvas
    new_img = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    new_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # Adjust for center of mass
    cy, cx = center_of_mass(new_img)
    
    if np.isnan(cx) or np.isnan(cy):
        return img
    
    shift_x = int(14 - cx)
    shift_y = int(14 - cy)
    
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    new_img = cv2.warpAffine(new_img, M, (28, 28))

    return new_img


uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

model = joblib.load("mnist_XGBoost_model.pkl")
scaler = joblib.load("mnist_scaler.pkl")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE) 

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)
    img = img[y:y+h, x:x+w]

    img = center_resize_digit(img)

    img = img.reshape(1, -1)
    img = scaler.transform(img)

    prediction = model.predict(img)[0]

    preproc_img = img.reshape(28, 28)
    if preproc_img.dtype != np.uint8:
        preproc_img = np.clip(preproc_img, 0, 1)
        preproc_img = (preproc_img * 255).astype(np.uint8)

    st.image(preproc_img, caption="Pre-processed picture", width=150)
    st.write(f"I think your number is: {prediction}")


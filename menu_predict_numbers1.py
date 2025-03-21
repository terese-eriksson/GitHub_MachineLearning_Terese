import streamlit as st
from streamlit_drawable_canvas import st_canvas
import joblib
import cv2
import numpy as np
from scipy.ndimage import center_of_mass

# Draw number
st.title("Draw a number")
st.write("Use the mouse to draw a number in the box below. Press the predict-button to get your prediction.")


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


canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

model = joblib.load("mnist_XGBoost_model.pkl")
scaler = joblib.load("mnist_scaler.pkl")

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    img = center_resize_digit(img)

    img = img.reshape(1, -1)

    img = scaler.transform(img)

    if st.button("Predict number"):
        prediction = model.predict(img)
        st.write(f"I think your number is: **{prediction[0]}**")
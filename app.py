import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load your YOLOv8 model
model = YOLO("best.pt")  # Update with your model path

# Constants
MAX_OXYGEN_KG = 118  # Max oxygen for mature tree (kg/year)

# Growth stage logic
def get_growth_stage(area):
    if area < 3000:
        return "Weak"
    elif area < 5000:
        return "Sapling"
    elif area < 15000:
        return "Young"
    elif area < 30000:
        return "Mature"
    else:
        return "Old"

# Estimate height from pixel height (assuming fixed pixel-to-meter ratio)
def estimate_height(pixel_height):
    pixel_to_meter_ratio = 90  # fixed ratio
    return round(pixel_height / pixel_to_meter_ratio, 2)

# Estimate oxygen production (kg/year)
def estimate_oxygen_output(height_m):
    return round(height_m * 11.8, 2)

# Estimate oxygen efficiency (% of max potential)
def oxygen_optimization(oxygen_kg):
    return round((oxygen_kg / MAX_OXYGEN_KG) * 100, 1)

# Streamlit UI
st.title("🌳 SUSTAINABLE MONITORING OF PUBLIC GREEN WEALTH")
st.write("Upload a tree image to predict tree growth stage, height, and estimate oxygen output & optimization potential.")

uploaded_file = st.file_uploader("Upload Tree Image", type=["jpg", "jpeg", "png"])

show_labels = st.checkbox("Show info on image", value=True)

if uploaded_file is not None:
    st.image(uploaded_file, caption="📷 Uploaded Image", use_column_width=True)
    st.write("⏳ Processing...")

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    results = model(img_array)[0]

    detection_info = []

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1
        height = y2 - y1
        area = width * height

        stage = get_growth_stage(area)
        height_m = estimate_height(height)
        oxygen_kg = estimate_oxygen_output(height_m)
        oxygen_pct = oxygen_optimization(oxygen_kg)

        # Draw bounding box and ID
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_array, f"#{i+1}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        if show_labels:
            label = f"{stage}, {height_m}m, {oxygen_kg}kg, {oxygen_pct}%"
            cv2.putText(img_array, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

        detection_info.append({
            "Tree #": i + 1,
            "Growth Stage": stage,
            "Height (m)": height_m,
            "O₂ Produced (kg/year)": oxygen_kg,
            "Optimization (%)": f"{oxygen_pct}%"
        })

    # Output image and table
    st.image(img_array, caption="🌱 Predicted Trees", use_column_width=True)

    if detection_info:
        st.write("### 📊 Tree Summary")
        st.dataframe(detection_info, use_container_width=True)
    else:
        st.warning("No trees detected.")

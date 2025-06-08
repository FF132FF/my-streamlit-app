import streamlit as st
from PIL import Image
import requests
import numpy as np
import io
from streamlit_drawable_canvas import st_canvas
import pandas as pd

st.title("Классификация изображений")

# Читаем URL из файла
#with open("/content/drive/MyDrive/api_project/public_url.txt", "r") as f:
#    public_url_str = f.read().strip()

api_url = f"https://b2cb-35-230-81-210.ngrok-free.app/predict/"

mode = st.radio("Выберите способ ввода изображения", ("Загрузить файл", "Нарисовать"))

image = None

if mode == "Загрузить файл":
    uploaded_file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif mode == "Нарисовать":
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=5,
        stroke_color="#000000",
        background_color="#eee",
        height=224,
        width=224,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        img_array = canvas_result.image_data.astype(np.uint8)
        if img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
        image = Image.fromarray(img_array).convert("RGB")

if image is not None:
    st.write(f"Тип image: {type(image)}")
    try:
        st.image(image, caption="Входное изображение", use_container_width=True)
    except Exception as e:
        st.error(f"Ошибка при отображении изображения: {e}")
    buffered = io.BytesIO()
    image = image.resize((224, 224))
    image.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
    try:
        response = requests.post(api_url, files=files)
        response.raise_for_status()
        data = response.json()

        st.write("### Результаты классификации")
        st.write(f"**Предсказанный класс:** {data['prediction']}")

        st.write("**Вероятности:**")
        probs = data["probabilities"]
        for cls, prob in probs.items():
            st.write(f"- {cls}: {prob:.4f}")

        df = pd.DataFrame.from_dict(probs, orient='index', columns=['Вероятность'])
        st.bar_chart(df)

    except requests.RequestException as e:
        st.error(f"Ошибка запроса: {e}")

"""
streamlit_app.py
Shareable Streamlit demo for FER-2013 emotion recognition.
Supports image upload and (optionally) live webcam via streamlit-webrtc.

Run:
    streamlit run streamlit_app.py
"""

import numpy as np
import streamlit as st
from PIL import Image
import cv2
from tensorflow.keras.models import load_model

EMOTION_LABELS  = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
EMOTION_EMOJIS  = ['😠', '🤢', '😨', '😄', '😐', '😢', '😲']
IMG_SIZE = 48

@st.cache_resource
def load_fer_model(path='models/best_model.h5'):
    return load_model(path)


def detect_and_predict(image_bgr, model, face_cascade):
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
    results = []
    for (x, y, w, h) in faces:
        roi  = gray[y:y+h, x:x+w]
        face = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        probs = model.predict(face.reshape(1, IMG_SIZE, IMG_SIZE, 1), verbose=0)[0]
        results.append({'box': (x, y, w, h), 'probs': probs})
    return results


def annotate_frame(image_bgr, results):
    out = image_bgr.copy()
    for r in results:
        x, y, w, h = r['box']
        idx   = np.argmax(r['probs'])
        label = f"{EMOTION_EMOJIS[idx]} {EMOTION_LABELS[idx]}  {r['probs'][idx]:.0%}"
        cv2.rectangle(out, (x, y), (x+w, y+h), (66, 153, 225), 2)
        cv2.putText(out, label, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (66, 153, 225), 2)
    return out


def main():
    st.set_page_config(page_title='FER Demo', page_icon='😄', layout='centered')
    st.title('😄 Facial Emotion Recognition')
    st.caption('CNN trained on FER-2013 · 7 emotion classes')

    try:
        model = load_fer_model()
    except Exception as e:
        st.error(f"Could not load model: {e}\nRun `python train.py` first.")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    uploaded = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded:
        pil_img   = Image.open(uploaded).convert('RGB')
        image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        results = detect_and_predict(image_bgr, model, face_cascade)

        if not results:
            st.warning("No face detected. Try a clearer frontal photo.")
            st.image(pil_img, use_column_width=True)
        else:
            annotated = annotate_frame(image_bgr, results)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)

            st.markdown(f"**{len(results)} face(s) detected**")
            for i, r in enumerate(results):
                st.markdown(f"#### Face {i+1}")
                probs = r['probs']
                for j, (lbl, emoji, p) in enumerate(zip(EMOTION_LABELS, EMOTION_EMOJIS, probs)):
                    st.progress(float(p), text=f"{emoji} {lbl}  {p:.1%}")


if __name__ == '__main__':
    main()

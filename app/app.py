# app.py  (updated)
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import io
import json
import os

st.set_page_config(page_title="Plant Disease Detector", layout="centered")

MODEL_PATH_DEFAULT = "trained_plant_disease_model.keras"

st.title("🌱 Plant Disease Detection")
st.write(
    """
Upload a plant image and the app will predict the disease class using a trained Keras model.
If you don't have a saved model in the working directory, you can upload one (.keras or .h5).
Optionally upload a `classes.txt` (one class per line) or `classes.json` (JSON array of strings)
to restore human-readable class names.
"""
)

# Lazy TensorFlow import: only import when actually loading a model.
@st.cache_resource
def load_model_from_path(path):
    """
    Tries to import tensorflow lazily and load the model.
    If TensorFlow is not installed, returns None and logs an informative message to the sidebar.
    """
    try:
        import tensorflow as tf
    except Exception as e:
        # TensorFlow isn't installed in this environment
        st.sidebar.error(
            "TensorFlow is not installed in this environment. "
            "To load a Keras model, install `tensorflow` or `tensorflow-cpu` in your environment "
            "(or upload a model-compatible file and run this locally)."
        )
        return None

    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.sidebar.error(f"Failed to load model from `{path}`: {e}")
        return None


def load_class_names_from_file(uploaded_file):
    if uploaded_file is None:
        return None
    # ensure we read from start
    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    name = uploaded_file.name.lower()
    # Try to decode as text first
    try:
        content_bytes = uploaded_file.read()
        if isinstance(content_bytes, bytes):
            content = content_bytes.decode("utf-8")
        else:
            # streamlit sometimes gives BytesIO; convert
            content = str(content_bytes)
    except Exception:
        # binary -> try json load directly from BytesIO (reset pointer)
        try:
            uploaded_file.seek(0)
            obj = json.load(uploaded_file)
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return obj
            return None
        except Exception:
            return None

    # try JSON array
    try:
        data = json.loads(content)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
    except Exception:
        pass

    # fallback: newline-separated text
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if lines:
        return lines
    return None


def _get_resample_filter():
    # Pillow 10+ uses Image.Resampling.LANCZOS; older versions use Image.ANTIALIAS
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    if hasattr(Image, "LANCZOS"):
        return Image.LANCZOS
    return Image.ANTIALIAS


def preprocess_image(img: Image.Image, img_size=(128, 128)):
    # ensure RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    # resize and scale
    resample = _get_resample_filter()
    img = ImageOps.fit(img, img_size, resample)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dimension
    return arr


def predict(model, image_array, top_k=5):
    # ensure model is callable
    preds = model.predict(image_array)
    if preds.ndim == 2:
        probs = preds[0]
    elif preds.ndim == 1:
        probs = preds
    else:
        probs = np.ravel(preds)
    top_idx = np.argsort(probs)[::-1][:top_k]
    top_probs = probs[top_idx]
    return top_idx, top_probs, probs


# Section: Model load / upload
st.sidebar.header("Model")
st.sidebar.write("Model will be loaded from the working directory if present.")
model_path_input = st.sidebar.text_input("Model path", value=MODEL_PATH_DEFAULT)
uploaded_model = st.sidebar.file_uploader(
    "Or upload a Keras model (.keras or .h5)", type=["keras", "h5", "hdf5"]
)

model = None
if uploaded_model is not None:
    # save uploaded model to temporary file and load
    temp_model_path = os.path.join(".", uploaded_model.name)
    with open(temp_model_path, "wb") as f:
        f.write(uploaded_model.getbuffer())
    st.sidebar.success(f"Saved uploaded model to {temp_model_path}")
    model = load_model_from_path(temp_model_path)
    if model is None:
        st.sidebar.error("Failed to load the uploaded model. See sidebar messages.")
else:
    # try to load default path if exists
    if os.path.exists(model_path_input):
        model = load_model_from_path(model_path_input)
        if model is None:
            st.sidebar.error(f"Found {model_path_input} but failed to load it. See sidebar messages.")
        else:
            st.sidebar.success(f"Loaded model from {model_path_input}")
    else:
        st.sidebar.info(f"No model found at {model_path_input}. You can upload one or place it in the working directory.")


# Section: class names upload
st.sidebar.header("Class Names (optional)")
classes_file = st.sidebar.file_uploader("Upload classes.txt or classes.json", type=["txt", "json"])
class_names = None
if classes_file is not None:
    try:
        classes_file.seek(0)
    except Exception:
        pass
    class_names = load_class_names_from_file(classes_file)
    if class_names:
        st.sidebar.success(f"Loaded {len(class_names)} class names.")
    else:
        st.sidebar.error("Could not parse class names file. Expect one class per line or a JSON array.")


# UI: image uploader
st.subheader("Upload an image to classify")
uploaded_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# optional image size control (should match training)
img_size = st.slider("Image size (square) — match your training size", min_value=32, max_value=512, value=128, step=8)

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)
    except Exception as e:
        st.error("Could not open the image. Make sure it's a valid image file.")
        image = None

    if image is not None:
        st.image(image, caption="Uploaded image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        if model is None:
            st.error("No model loaded. Upload a saved Keras model or place it in the working directory. "
                     "If TensorFlow is not installed, install `tensorflow` or run locally.")
        else:
            # preprocess and predict
            input_arr = preprocess_image(image, img_size=(img_size, img_size))
            try:
                top_idx, top_probs, all_probs = predict(model, input_arr, top_k=5)
            except Exception as e:
                st.error(f"Prediction error: {e}")
                top_idx, top_probs, all_probs = None, None, None

            if top_idx is not None:
                # human-readable labels if available
                if class_names and len(class_names) > max(top_idx):
                    readable = [class_names[i] for i in top_idx]
                else:
                    readable = [f"Class {i}" for i in top_idx]

                # Display top result
                st.markdown("### ✅ Prediction")
                st.write(f"**Top predicted class:** `{readable[0]}` — **{top_probs[0]*100:.2f}%** confidence")

                # Show top-k as table
                rows = []
                for r_i, (idx, prob, name) in enumerate(zip(top_idx, top_probs, readable)):
                    rows.append({"rank": r_i+1, "class_index": int(idx), "class_name": name, "probability": float(prob)})
                import pandas as pd
                df = pd.DataFrame(rows)
                st.table(df)

                # bar chart of top-k
                try:
                    st.subheader("Top probabilities")
                    st.bar_chart(pd.DataFrame({"probability": top_probs}, index=readable))
                except Exception:
                    pass

                # Option to download raw probabilities
                try:
                    st.download_button("Download full probabilities (JSON)",
                                       data=json.dumps(all_probs.tolist()),
                                       file_name="probs.json",
                                       mime="application/json")
                except Exception:
                    # some models might return non-serializable types
                    st.info("Unable to create downloadable probabilities file for this model.")
else:
    st.info("Upload an image to get started. If you want to test quickly, upload both a model (.keras/.h5) and a class names file (classes.txt).")


# Footer: helpful tips
st.markdown("---")
st.markdown(
    """
**Tips**
- The model used in the original notebook saved as `trained_plant_disease_model.keras` (place that file in this folder).  
- If you trained with 128×128 images, set the Image size slider to 128.  
- If predictions look wrong, check class ordering — the class names file must match the model's training label order.
"""
)

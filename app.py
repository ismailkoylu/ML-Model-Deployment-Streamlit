import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np

@st.cache_resource
def load_caption_model():
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_caption_model()


# set title and sidebar

st.title("Image Captioner")
st.sidebar.text("İsmail Köylü - Data Science Portfolio")
st.sidebar.header("Parameter Settings")

# set sidebar sliders



temperature = st.sidebar.slider(
    label="Temperature",
    min_value=0.1,
    max_value=1.5,
    value=1.0,
    step=0.1
)



max_length = st.sidebar.slider(
    label="Max Length",
    min_value=5,
    max_value=30,
    value=20,
    step=1
)



min_length = st.sidebar.slider(
    label="Min Length",
    min_value=3,
    max_value=20,
    value=5,
    step=1
)



num_captions = st.sidebar.slider(
    label="Number of Variations",
    min_value=1,
    max_value=5,
    value=1,
    step=1
)

# STEP 3: MAIN PIPELINE

st.write("Upload an image to test my AI!")

# File uploader for images, accepts jpg, jpeg, png

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:
    
    raw_image = Image.open(uploaded_file).convert('RGB')
    st.image(raw_image, caption='Uploaded Image', use_container_width=True)

    
    text_input = st.text_input("Start the sentence with: (Optional)", placeholder="Eg: A photo of a...")

    
    if st.button("Generate Caption"):
        
        with st.spinner("AI is checking your image, be patient!"):

            st.subheader("Caption(s):")
            # PRE-PROCESSING (Convert Image & Text to Tensors) ---
            if text_input:
                # If user enters a starting text
                inputs = processor(images=raw_image, text=text_input, return_tensors="pt")
            else:
                # If we have only image
                inputs = processor(images=raw_image, return_tensors="pt")


            
            for i in range(num_captions):
                
                out = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    max_length=max_length,
                    min_length=min_length,
                    top_k=50
                )

                # POST-PROCESSING (Decode Tensors back to Text)
            
                caption = processor.decode(out[0], skip_special_tokens=True)

                st.success(f"Caption {i + 1}: {caption}")
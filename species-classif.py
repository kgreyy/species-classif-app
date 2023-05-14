import streamlit as st
import cv2
from PIL import Image
import numpy as np
import base64

import argparse
# import jax
import json
import matplotlib.pyplot as plt
import os
import pandas as pd

from PIL import Image
# from transformers import CLIPProcessor, FlaxCLIPModel


@st.cache_resource()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

def predict_one_image(image_file, model, processor, class_names, k, order):
    eval_image = Image.fromarray(plt.imread(os.path.join(IMAGES_DIR, image_file)))
    eval_sentences = ["Belongs to {:s} {:s}".format(order, ct) for ct in class_names]
    inputs = processor(text=eval_sentences,
                       images=eval_image,
                       return_tensors="jax",
                       padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = jax.nn.softmax(logits_per_image, axis=-1)
    probs_np = np.asarray(probs)[0]
    probs_npi = np.argsort(-probs_np)
    predictions = [(class_names[i], probs_np[i]) for i in probs_npi[0:k]]
    return predictions

DATA_DIR = "data/"
# IMAGES_DIR = os.path.join(DATA_DIR, "RSICD_images")
IMAGES_DIR = os.path.join(DATA_DIR)

BASELINE_MODEL = "openai/clip-vit-base-patch32"
MODEL_DIR = "/home/shared/models/clip-rsicd"
K_VALUES = [1, 3, 5, 10]


if __name__=='__main__':
    set_png_as_page_bg('white-concrete-wall.jpg')
    df_categs = pd.read_csv("data/test_hierarchy_reference.csv")

    st.markdown('<h1 style="color:black;">CLIP Species Classification model</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:gray;">The image classification model classifies images according to:</h2>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:gray;">'+', '.join(df_categs.columns)+'</h3>', unsafe_allow_html=True)

    

    class_col = st.selectbox('Select what to classify on', df_categs.columns, len(df_categs.columns)-1)
    st.markdown('### You have selected ' + class_col)
    st.markdown('<h3 style="color:black;"> There are '+str(len(df_categs[class_col].unique()))+' classes of taxonomic classification: '+class_col+'.</h3>', unsafe_allow_html=True)

    # Image upload
    upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
    c1, c2= st.columns(2)
    if upload is not None:
        im= Image.open(upload)
        img= np.asarray(im)
        image= cv2.resize(img,(224, 224))
        img= np.expand_dims(img, 0)
        c1.header('Input Image')
        c1.image(im)
        c1.write(img.shape)
        model = FlaxCLIPModel.from_pretrained(BASELINE_MODEL)
        processor = CLIPProcessor.from_pretrained(BASELINE_MODEL)
        class_names = df_categs[class_col].unique()
        preds = predict_one_image(
                img, model, processor, class_names, max(K_VALUES), class_col)
        st.markdown('<h3 style="color:black;"> The model predicted: '+'\n'.join(['\t'.join(r) for r in preds])+'</h3>')
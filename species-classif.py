import streamlit as st
import cv2
from PIL import Image
import numpy as np
import base64

import argparse
import jax
import json
import matplotlib.pyplot as plt
import os
import pandas as pd

from PIL import Image
from transformers import CLIPProcessor, FlaxCLIPModel


@st.cache_resource()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_page_bg():
    page_bg_img = '''
    <style>
    .stApp {
    background-color: white;
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    '''
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

def predict_one_image(image_file, model, processor, class_names, k, order):
    eval_image = image_file
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

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


if __name__=='__main__':
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

    set_page_bg()
    df_categs = pd.read_csv("data/test_hierarchy_reference.csv")

    st.markdown('# CLIP Taxon Classification model')
    st.markdown('## The image classification model classifies images according to:')
    st.markdown('### '+', '.join(df_categs.columns))

    model_map = {"openai/clip-vit-base-patch32": 'baseline'}
    model_name = st.selectbox('Select what model to use', ["openai/clip-vit-base-patch32", 'model12/',
                                                           'model18/', 'model20/', 'model27/'],
                              0, format_func=lambda x: model_map.get(x, x[:-1]))

    class_col = st.selectbox('Select what to classify on', df_categs.columns, len(df_categs.columns)-1)
    st.markdown('### You have selected ' + class_col)

    phylum_hint = st.selectbox('Hint a phylum where the species is under', ['.+']+sorted(list(df_categs['phylum'].dropna().unique())),
                               0, format_func=lambda x: {'.+':'No hint'}.get(x, x))
    
    filtered_df = df_categs.loc[df_categs['phylum'].str.match(phylum_hint, na=False)]

    class_hint = st.selectbox('Hint a class where the species is under', ['.+']+sorted(list(filtered_df['class'].dropna().unique())),
                               0, format_func=lambda x: {'.+':'No hint'}.get(x, x))
    
    filtered_df = filtered_df.loc[filtered_df['class'].str.match(class_hint, na=False)]
    
    family_hint = st.selectbox('Hint a family where the species is under', ['.+']+sorted(list(filtered_df['family'].dropna().unique())),
                               0, format_func=lambda x: {'.+':'No hint'}.get(x, x))

    class_names = filtered_df.loc[filtered_df['family'].str.match(family_hint, na=False), class_col].dropna().unique()
    
    st.markdown('### There are '+str(len(class_names))+' classes of taxonomic classification: '+class_col)

    # Image upload
    upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
    c1, c2= st.columns(2)

    
    if upload is not None and st.button("Predict"):
        im = Image.open(upload)
        img = np.asarray(im)
        img = image_resize(img, height = 720)
        img = np.expand_dims(img, 0)
        c1.header('Input Image')
        c1.image(im)
        c1.write(img.shape)
        model = FlaxCLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(BASELINE_MODEL)
        
        preds = predict_one_image(
                img, model, processor, class_names, max(K_VALUES), class_col)
        st.markdown('The model predicted:',unsafe_allow_html=True)
        df = pd.DataFrame(preds, columns=['pred', 'probability'])
        st.table(df)

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

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file) 
    page_bg_img = '''
    <style>
    .stApp {
    background-color: black;
    /*background-image: url("data:image/png;base64,%s");*/
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str
    
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

    set_png_as_page_bg('white-concrete-wall.jpg')
    df_categs = pd.read_csv("data/test_hierarchy_reference.csv")

    st.markdown('<h1 style="color:white;">CLIP Taxon Classification model</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:gray;">The image classification model classifies images according to:</h2>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:gray;">'+', '.join(df_categs.columns)+'</h3>', unsafe_allow_html=True)

    

    class_col = st.selectbox('Select what to classify on', df_categs.columns, len(df_categs.columns)-1)
    st.markdown('<h3 style="color:gray;">You have selected ' + class_col +'.</h3>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:gray;">There are '+str(len(df_categs[class_col].unique()))+' classes of taxonomic classification: '+class_col+'.</h3>', unsafe_allow_html=True)

    # Image upload
    upload= st.file_uploader('Insert image for classification', type=['png','jpg'])
    c1, c2= st.columns(2)
    if upload is not None:
        im = Image.open(upload)
        img = np.asarray(im)
        img = image_resize(img, height = 720)
        img = np.expand_dims(img, 0)
        c1.header('Input Image')
        c1.image(im)
        c1.write(img.shape)
        model = FlaxCLIPModel.from_pretrained(BASELINE_MODEL)
        processor = CLIPProcessor.from_pretrained(BASELINE_MODEL)
        class_names = df_categs[class_col].dropna().unique()
        preds = predict_one_image(
                img, model, processor, class_names, max(K_VALUES), class_col)
        st.markdown('<h3 style="color:white;"> The model predicted: </h3>',unsafe_allow_html=True)
        df = pd.DataFrame(preds, columns=['pred', 'probability'])
        st.table(df)

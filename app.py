#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 17:55:52 2020

@author: raghu
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st
import os
import sys



def main():

    st.set_option('deprecation.showfileUploaderEncoding', False)
    #Building sidebar of web app
    st.write("""
    # LC50 predictor
    ***Lethal concentration 50*** (**LC50**) is the amount of a substance suspended
     in the water required to kills 50% of a test fish during a predetermined
     observation period. **LC50** values are frequently used as a general indicator
     of a substance's acute toxicity.
    """)

    st.sidebar.header('User Input Parameters')

    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV or Excel file", type=['csv', 'xls', 'xlsx'])
    st.sidebar.markdown("<p style='text-align: center; color: black;'>-- or --</p>", unsafe_allow_html=True)

    if uploaded_file is not None:
        #os.system('python maincode.py') #comment out this line if not needed
        def try_read_df(f):
            try:
                return pd.read_csv(f)
            except:
                return pd.read_excel(f)
        input_df = try_read_df(uploaded_file)
        if len(input_df.columns.values) != 6:
            st.subheader('Input file does not have 6 features, please try again')
            return None
        if 'object' in input_df.dtypes.values:
            st.subheader('Input file has non-numeric values, please try again.')
            return None
        #input_df = pd.read_csv(uploaded_file)
    else:
        #os.system('python maincode.py') #comment out this line if not needed
        def user_input_features():
            feature1 = st.sidebar.number_input("CIC0", value = 2.94)
            feature2 = st.sidebar.number_input("SM1_Dz(Z)", value = 0.56)
            feature3 = st.sidebar.number_input("GATS1i", value = 1.23)
            feature4 = st.sidebar.slider('NdsCH',  0, 4, 0, step = 1)
            feature5 = st.sidebar.slider('NdssC', 0, 6, 0, step = 1)
            feature6 = st.sidebar.number_input("MLOGP", value = 2.13)

            dataIn = {'CIC0': feature1,
                    'SM1_Dz(Z)': feature2,
                    'GATS1i': feature3,
                    'NdsCH': feature4,
                    'NdssC': feature5,
                    'MLOGP': feature6}
            features = pd.DataFrame(dataIn, index=[0])
            return features
        input_df = user_input_features()

    inputVals = input_df
    toPredict = inputVals.copy()

    continuous = ['CIC0', 'SM1_Dz(Z)', 'GATS1i', 'MLOGP']
    skewedFeatures = ['SM1_Dz(Z)', 'GATS1i']
    #scale
    scaleSaved = pickle.load(open('scale.pickle', 'rb'))
    toPredict[continuous] = scaleSaved.transform(toPredict[continuous])
    #transform
    toPredict[skewedFeatures] = np.sqrt(toPredict[skewedFeatures])
    #dimensionality reduction
    pca = pickle.load(open('pca.pickle', 'rb'))
    toPredict = pca.transform(toPredict)
    #predict
    filename = 'finalized_model.pickle'
    loaded_model = pickle.load(open(filename, 'rb'))
    a = loaded_model.predict(toPredict)
    inputVals['LC50'] = a

    st.subheader('User Input parameters and predictions')
    st.write(inputVals)

    nameHandle = open("logs/ProcessLog.txt", 'r')
    #st.write(*nameHandle.readlines())
    pLog = ''
    for line in nameHandle.readlines():
        pLog += line

    st.subheader('Process Log')
    pLog = '<div style="height:300px;width:700px;border:1px solid #ccc;font:16px/26px Georgia, Garamond, Serif;overflow:auto;">'+pLog+'</div>'
    st.markdown(pLog, unsafe_allow_html=True)

if __name__=='__main__':
    main()

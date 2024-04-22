import streamlit as st
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage.filters import sobel

# Load models and pre-trained objects
@st.cache(allow_output_mutation=True)
def load_models():
    # Load VGG16 model for feature extraction
    model_vgg = VGG16()
    model_vgg = Model(inputs=model_vgg.inputs, outputs=model_vgg.layers[-2].output)

    # Load SVM model and label encoder
    with open('SVM.pkl', 'rb') as file:
        svm_model = pickle.load(file)

    with open('label_encoder.pkl', 'rb') as file:
        le = pickle.load(file)

    # Load PCA model
    with open('pca_model.pkl', 'rb') as file:
        pca = pickle.load(file)

    return model_vgg, svm_model, le, pca

model_vgg, svm_model, le, pca = load_models()

# Function to extract features from an image
def extract_features(image_path, model):
    features = {}
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = preprocess_input(img_array)
        feature = model.predict(img_array, verbose=0)
        features[image_path] = feature
    except Exception as e:
        st.error(f"Error processing image: {e}")
    return features

# Function to extract HOG features and perform edge detection on an image
def extract_hog_and_edge(image):
    # Convert image to grayscale
    gray_image = image.convert('L')
    
    # Extract HOG features
    hog_features, hog_image = hog(np.array(gray_image), orientations=10, pixels_per_cell=(12, 12),
                                   cells_per_block=(1, 1), visualize=True)
    
    # Normalize the HOG image array
    hog_image = (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min())
    
    # Perform edge detection using the Sobel operator
    edge_image = sobel(np.array(gray_image))
    
    return hog_image, edge_image

# UI
st.title('Fashion Image Classifier')

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Get the location (path) of the uploaded file
    image_location = uploaded_file.name
    
    # Extract features from the uploaded image
    features = extract_features(image_location, model_vgg)

    if features:
        st.write("Features extracted successfully.")
        
        if st.button('Predict'):
            try:
                # Transform features using PCA
                test = pca.transform(features[image_location])
                
                # Make prediction using SVM model
                predicted_label = svm_model.predict(test)
                
                # Decode predicted label
                decoded_label = le.inverse_transform(predicted_label)
                
                # Display prediction
                st.success(f'Prediction: {decoded_label[0]}')
            except Exception as e:
                st.error(f"Error predicting: {e}")
    
    # Display buttons for HOG image, edge detection image, and prediction
    col1, col2, col3 = st.columns(3)
    show_hog = col1.button('HOG Image')
    show_edge = col2.button('Edge Detection Image')
    show_prediction = col3.button('Prediction')

    hog_image, edge_image = None, None

    if show_hog:
        hog_image, _ = extract_hog_and_edge(image)

    if show_edge:
        _, edge_image = extract_hog_and_edge(image)

    if show_hog or show_edge:
        col1, col2 = st.columns(2)
        if hog_image is not None:
            col1.image(hog_image, caption='HOG Image', use_column_width=True)
        if edge_image is not None:
            col2.image(edge_image, caption='Edge Detection Image', use_column_width=True)

    if show_prediction:
        if features:
            try:
                # Transform features using PCA
                test = pca.transform(features[image_location])
                
                # Make prediction using SVM model
                predicted_label = svm_model.predict(test)
                
                # Decode predicted label
                decoded_label = le.inverse_transform(predicted_label)
                
                # Display prediction
                st.success(f'Prediction: {decoded_label[0]}')
            except Exception as e:
                st.error(f"Error predicting: {e}")
        else:
            st.warning("No features extracted.")

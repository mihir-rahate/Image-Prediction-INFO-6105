import streamlit as st
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
        
# Load the VGG16 model
model_vgg = VGG16()
# Restructure the model to just extract the features by removing the prediction layer
model_vgg = Model(inputs=model_vgg.inputs, outputs=model_vgg.layers[-2].output)

# Load the SVM model
with open('SVM.pkl', 'rb') as file:
    svm_model = pickle.load(file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

with open('pca_model.pkl', 'rb') as file:
    pca = pickle.load(file)



st.title('Image Classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image_location = uploaded_file.name
    print("Location of the uploaded image:", image_location)
    # Extract features
    try:
        # Load the image with a target size
        img = load_img(uploaded_file, target_size=(224, 224))
        
        # Convert the image to a numpy array
        img_array = img_to_array(img)
        
        # Reshape the data for the model
        img_array = np.expand_dims(img_array, axis=0)  # This adds a batch dimension
        
        # Preprocess the image array
        img_array = preprocess_input(img_array)
        
        # Extract features using the pretrained model
        processed_input = model_vgg.predict(img_array, verbose=0)
        
        # Check the shape of the features
        print('Feature shape:', processed_input.shape)
        
        def load_sample_images(image_path,  model):
            features = {}
            try:
                img = load_img(image_path, target_size=(224, 224))
                img_array = img_to_array(img)
                img_array = img_array.reshape((1, *img_array.shape))
                img_array = preprocess_input(img_array)
                feature = model.predict(img_array, verbose=0)
                features[image_path] = feature
            except Exception as e:
                print(f"Error processing image: {e}")
            return features

        features = load_sample_images(image_location, model_vgg)
        features
        print(features)
        print(image_location)

        print('4')
        test=pca.transform(features[image_location])
        print('2')
        #print("Shape after PCA:", features.shape)  # Should be (1, 100)
        print('1')
        if st.button('Predict'):
            # Make prediction
            predicted_label = svm_model.predict(test)
            decoded_label = le.inverse_transform(predicted_label)
            print(predicted_label)
            st.write(f'Prediction: {predicted_label}')
            st.write(f'Prediction: {decoded_label[0]}')

    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")

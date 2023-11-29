#!/usr/bin/env python
# coding: utf-8

# ## Transfer Learning for Image Classification

# #### Unzipping the dataset

# In[1]:


#Necessary libraries for modelling
get_ipython().system('pip install --upgrade tensorflow')
get_ipython().system('pip install --upgrade keras')
get_ipython().system('pip install Pillow')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install tabulate')


# In[2]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[3]:


import pandas as pd


# In[1]:


import os

# Get the current working directory
current_directory = os.getcwd()

# Print the current directory
print("Current Directory:", current_directory)


# In[2]:


import os
import zipfile

def unzip_nested_zip(zip_file_path, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)
    
    print(f"Unzipped '{zip_file_path}' to '{output_folder}'.")
    
    # Check for nested zip files and unzip them recursively
    extracted_files = os.listdir(output_folder)
    for extracted_file in extracted_files:
        extracted_file_path = os.path.join(output_folder, extracted_file)
        if extracted_file_path.endswith('.zip') and zipfile.is_zipfile(extracted_file_path):
            # Recursive call to handle nested zip files
            unzip_nested_zip(extracted_file_path, os.path.splitext(extracted_file_path)[0])


# In[3]:


# Specify the path to the initial zip folder
initial_zip_folder_path = "/tf/trailanderror/yogapose/Yoga Pose.zip"

# Specify the output folder where contents will be extracted
output_folder = "/tf/trailanderror/yogapose"

# Call the function to recursively unzip folders within folders and nested zip files
unzip_nested_zip(initial_zip_folder_path, output_folder)


# In[4]:


from PIL import Image


# In[5]:


images = []
labels = []

master_data_path="/tf/trailanderror/yogapose"

def load_images_from_folder(folder_path, label):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path):
            # If the item is a directory, recursively load images from it
            load_images_from_folder(file_path, filename)
        elif filename.lower().endswith(('png', 'jpg', 'jpeg')):
            # If the item is an image file, load it and add it to the images list
            try:
                img = Image.open(file_path)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f'Error loading image {file_path}: {str(e)}')
        else:
            print(f'Skipping non-image file: {file_path}')

# Iterate through folders in the master folder
for folder_name in os.listdir(master_data_path):
    folder_path = os.path.join(master_data_path, folder_name)
    if os.path.isdir(folder_path):
        # Load images and labels from the current folder
        load_images_from_folder(folder_path, folder_name)

print(f'Images loaded: {len(images)}')
print(f'Labels loaded: {len(labels)}')


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


# Iterate through the first two elements of labels and images
for label, image in zip(labels[0:2], images[0:2]):
    print(label)
    plt.imshow(image)
    plt.show()


# In[8]:


unique_labels = list(set(labels))
print(unique_labels)


# In[9]:


#counting the number of labels in each classes

count_of_classes = {}

for label in labels:
    if label in count_of_classes:
        count_of_classes[label] +=1
    else:
        count_of_classes[label] = 1
        
        
for key, value in count_of_classes.items():
    print(f'{key}:{value}')


# ### Data Processing

# In[10]:


Dimensions = []

for idx, img in enumerate(images):
    width, height = img.size
    current_dimension = (width, height)
    Dimensions.append(current_dimension)

unique_dimension_count = len(list(set(Dimensions)))

print(f'we have images with {unique_dimension_count} various dimensions')


# #### Image :- we have to convert all the images to have same dimesion and normalize them

# In[11]:


import numpy as np


# In[12]:


# Convert PIL images to numpy arrays
numpy_images = [np.array(image) for image in images]

# Resize and convert images to RGB format if necessary
target_size = (224, 224)
reshaped_images = []
for idx, image in enumerate(numpy_images):
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize(target_size)
    # Convert to RGB if image is grayscale
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    reshaped_images.append(np.array(pil_image))

# Shape of reshaped images
for idx, img in enumerate(reshaped_images):
    print(f'Image {idx+1} shape after resizing {img.shape}')


# In[13]:


Dimensions = []

for idx, img in enumerate(reshaped_images):
    width, height, channel = img.shape
    current_dimension = width, height, channel
    Dimensions.append(current_dimension)
    
unique_dimension_count_after_reshaping = len(list(set(Dimensions)))

print(f'Number of unique dimension: {unique_dimension_count_after_reshaping}')

print("Dimension of reshaped_images:", reshaped_images[0].shape)

print("Data type of reshaped_images:", reshaped_images[0].dtype)
                    


# In[14]:


# Convert images to float32 and normalize to [0, 1]
normalized_images = np.array(reshaped_images, dtype=np.float32) / 255.0

# Verify the shape and data type of processed_images
print("Shape of normalized_images:", normalized_images.shape)
print("Data type of processed_images:", normalized_images.dtype)


# In[15]:


print(labels[0])
plt.imshow(normalized_images[0])
plt.show()


# ## Handling Data imbalance :- 
# 
# The data is significantly imbalanced, so data balancing is essential to prevent the model from biassing towards ships.

# In[16]:


#counting the number of labels in each classes

count_of_classes = {}

for label in labels:
    if label in count_of_classes:
        count_of_classes[label] +=1
    else:
        count_of_classes[label] = 1
        
        
for key, value in count_of_classes.items():
    print(f'{key}:{value}')


# In[17]:


required_size = round(len(labels)/2)
print(f"we should have {required_size} images in each class")


# ###### The data requires balancing but it requires more time, so ignoring it as of now!!

# In[18]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

# Initialize empty lists to store processed images and corresponding labels
processed_images = []
processed_labels = []

# Load, preprocess, and align images and labels
for image, label in zip(normalized_images, labels):
    try:
        processed_images.append(image)
        processed_labels.append(label)
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")

# Convert labels to one-hot encoding for 107 classes
label_encoder = LabelEncoder()
integer_encoded_labels = label_encoder.fit_transform(processed_labels)
# Note: integer_encoded_labels now contains integers from 0 to 106 for 107 classes

onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
onehot_encoded_labels = onehot_encoder.fit_transform(integer_encoded_labels.reshape(-1, 1))

# Convert processed_images and onehot_encoded_labels to numpy arrays
processed_images = np.array(processed_images)
onehot_encoded_labels = np.array(onehot_encoded_labels)

# Save processed_images and onehot_encoded_labels in the current directory
np.save("processed_images.npy", processed_images)
np.save("onehot_encoded_labels.npy", onehot_encoded_labels)

# Verify the shapes of processed_images and onehot_encoded_labels
print("Shape of processed_images:", processed_images.shape)
print("Shape of onehot_encoded_labels:", onehot_encoded_labels.shape)


# # Model Selection and Transfer Learning

# ## Base Models

# In[19]:


# Import ResNeXt50 from tensorflow.keras.applications
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, InceptionV3, DenseNet121, MobileNetV2


# In[20]:


# ResNeXt50, SEResNet50 ; we have to manually import as they not available directly through tf.keras.applications 
# or import from torch #

#importing the model & dense layer for customizing the neural network
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers


# In[21]:


#input shape & dimension for the pre-trained models
shape=(224, 224, 3)


# In[22]:


# Load pre-trained models
base_model_1 = VGG16(weights='imagenet', include_top=False, input_shape=shape)
base_model_2 = VGG19(weights='imagenet', include_top=False, input_shape=shape)
base_model_3 = ResNet50(weights='imagenet', include_top=False, input_shape=shape)
base_model_4 = InceptionV3(weights='imagenet', include_top=False, input_shape=shape)
base_model_5 = DenseNet121(weights='imagenet', include_top=False, input_shape=shape)
base_model_6 = MobileNetV2(weights='imagenet', include_top=False, input_shape=shape)


# In[23]:


base_models = [base_model_1, base_model_2, base_model_3, base_model_4, base_model_5, base_model_6]


# In[24]:


# Looping through the Base models and printing the summaries
for idx, model in enumerate(base_models):
    print(f'Summary of Base Model {idx +1}:')
    model.summary()


# In[25]:


#Freezing the pre-trained model's last layer for transfer learning
for model in base_models:
    for layer in model.layers:
        layer.trainable=False


# ## Customize & Compile the Base Models

# In[26]:


base_models = [base_model_1, base_model_2, base_model_3, base_model_4, base_model_5, base_model_6]

custom_models = []

for idx, model in enumerate(base_models):
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(107, activation='softmax')(x)
    custom_model = Model(inputs=model.input, outputs=predictions)
    custom_models.append(custom_model)
    print(f"Customized the model - {idx+1}")


# In[27]:


# Looping through the Base models and printing the summaries
for idx, model in enumerate(custom_models):
    print(f'Summary of Custom Model {idx +1}:')
    model.summary()


# In[28]:


compiled_models = []
for idx, custom_model in enumerate(custom_models):
    custom_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    compiled_models.append(custom_model)
    print(f"Compiled Custom Model {idx + 1}")


# ## Train the model

# In[29]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, ZeroPadding2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

from tensorflow.keras.optimizers import Adam, SGD, RMSprop


# In[31]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(processed_images, onehot_encoded_labels, test_size=0.2, random_state=42)

#no validate split as of now

# Verify the shapes of the training and testing sets
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)


# In[41]:


model_names = ["VGG16", "VGG19", "ResNet50", "InceptionV3", "DenseNet121", "MobileNetV2"]
class_labels = ['matsyasana', 'uttana shishosana', 'ustrasana', 'tolasana', 'marichyasana i', 'vriksasana', 'eka pada koundinyanasana i', 'setu bandha sarvangasana', 'parivrtta parsvakonasana', 'garudasana', 'utkatasana', 'hanumanasana', 'pincha mayurasana', 'supta padangusthasana', 'adho mukha vriksasana', 'krounchasana', 'marichyasana iii', 'ardha matsyendrasana', 'virabhadrasana ii', 'natarajasana', 'anantasana', 'camatkarasana', 'ardha pincha mayurasana', 'paripurna navasana', 'urdhva mukha svanasana', 'tulasana', 'bakasana', 'chakravakasana', 'urdhva dhanurasana', 'eka pada rajakapotasana', 'ananda balasana', 'mayurasana', 'ardha bhekasana', 'bitilasana', 'kurmasana', 'padangusthasana', 'parighasana', 'eka pada rajakapotasana ii', 'tittibhasana', 'tadasana', 'balasana', 'virabhadrasana i', 'dandasana', 'bhujangasana', 'adho mukha svanasana', 'ganda bherundasana', 'utthita trikonasana', 'virabhadrasana iii', 'janu sirsasana', 'uttanasana', 'salamba bhujangasana', 'halasana', 'anjaneyasana', 'ardha chandrasana', 'malasana', 'bhairavasana', 'ardha uttanasana', 'vrischikasana', 'durvasasana', 'parsva bakasana', 'utthita ashwa sanchalanasana', 'supta baddha konasana', 'purvottanasana', 'prasarita padottanasana', 'marjaryasana', 'astavakrasana', 'bhujapidasana', 'bharadvajasana i', 'vasisthasana', 'parsvottanasana', 'chaturanga dandasana', 'dhanurasana', 'bhekasana', 'upavistha konasana', 'salamba sirsasana', 'paschimottanasana', 'savasana', 'agnistambhasana', 'dwi pada viparita dandasana', 'supta matsyendrasana', 'ashtanga namaskara', 'makarasana', 'lolasana', 'padmasana', 'urdhva hastasana', 'pasasana', 'urdhva prasarita eka padasana', 'utthita hasta padangustasana', 'baddha konasana', 'sukhasana', 'virasana', 'kapotasana', 'parivrtta janu sirsasana', 'simhasana', 'yoganidrasana', 'utthita parsvakonasana', 'gomukhasana', 'phalakasana', 'garbha pindasana', 'eka pada koundinyanasana ii', 'supta virasana', 'salamba sarvangasana', 'parivrtta trikonasana', 'vajrasana', 'viparita karani', 'salabhasana', 'makara adho mukha svanasana']


# In[42]:


def train_and_evaluate_models(X_train, y_train, X_test, y_test, compiled_models, model_names, epochs=100):
    results = []
    for model, model_name in zip(compiled_models, model_names):
        # Initialize variables for tracking maximum accuracy
        max_accuracy = 0
        max_accuracy_epoch = 0

        # Define a checkpoint to save the model when target accuracy is reached
        checkpoint = ModelCheckpoint(f'{model_name}_model_FR.h5', monitor='val_accuracy', 
                                     save_best_only=True, save_weights_only=False, mode='max', verbose=1)

        # Define early stopping to stop training if accuracy doesn't improve for 10 epochs
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=2, verbose=1)

        # Train the current model 
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=32, callbacks=[checkpoint, early_stopping])
        
        
        current_epoch_predictions = []
        current_epoch_labels = []
        
        for epoch, val_accuracy in enumerate(history.history['val_accuracy'], 1):
            # Check if the current epoch achieves higher accuracy than the previous maximum
            if val_accuracy > max_accuracy:
                max_accuracy = val_accuracy
                max_accuracy_epoch = epoch
                current_epoch_predictions = model.predict(X_test)
                current_epoch_labels = y_test.argmax(axis=1)

        # Get the maximum accuracy and the corresponding epoch
        max_accuracy_epoch = np.argmax(history.history['val_accuracy'])
        max_accuracy = history.history['val_accuracy'][max_accuracy_epoch]

        # Evaluate the model on the test data
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f'Test Accuracy for {model_name}: {test_accuracy}')

        # Store results
        results.append({'Model': model_name, 'Accuracy': max_accuracy})

        print("-" * 40)  # Print a line of dashes 
        print(f"Maximum accuracy of {max_accuracy:.2f} achieved at epoch {max_accuracy_epoch+1}")
        print("Model with high accuracy is saved using the keras ModelCheckpoint")
        print("-" * 40)  # Print a line of dashes 
        print("\n")

        # Assuming model.predict returns probabilities for each class
        y_pred_probs = model.predict(X_test)

        # Convert probabilities to class labels
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Convert true labels to class labels if y_test is one-hot encoded
        y_true = np.argmax(y_test, axis=1)

    return results


# In[43]:


#Train and Evaluate the model
results = train_and_evaluate_models(X_train, y_train, X_test, y_test, compiled_models, model_names, epochs=100)


# In[45]:


import pandas as pd
results_df_fr = pd.DataFrame(results)
# Print results in tabular form
print(results_df_fr)


# In[48]:


from tabulate import tabulate


# Define the headers for the table
headers = ["Model", "Test Accuracy"]

# Print the table
print('Transfer Learning - UnBalanced Data :- 107 Classes')
print(tabulate(results_df_fr, headers, tablefmt="grid"))


# ## The data is unbalanced and minimal, so in order to improve the accuracy of the models, we have to generate synthetic data and there is need to balance the data.  The balanced and improved model codes are also uploaded in the Github Respository.

# # Customizing the resnet model with various hyper parameters

# In[ ]:


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


# In[53]:


# Load ResNet-50 model with pre-trained weights
resnet_50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze layers in the pre-trained model
for layer in resnet_50.layers:
    layer.trainable = False

# List of parameters to iterate over
learning_rates = [0.0001, 0.001, 0.01, 0.0003, 0.003, 0.03]
loss_functions = ['categorical_crossentropy', 'mean_squared_error']
optimizers = [Adam, SGD, RMSprop]
dropout_rates = [0, 0.1, 0.3, 0.5]
batch_sizes = [8, 16, 32, 64, 128]

compiled_models_resnet = []
model_names_resnet = []

# Loop through learning rates, loss functions, optimizers, dropout rates, and batch sizes
for lr in learning_rates:
    for loss_function in loss_functions:
        for optimizer_class in optimizers:
            for dropout_rate in dropout_rates:
                for batch_size in batch_sizes:
                    # Create a new custom model
                    x = resnet_50.output
                    x = Flatten()(x)
                    x = Dense(1024, activation='relu')(x)
                    x = Dropout(dropout_rate)(x)
                    x = BatchNormalization()(x)
                    predictions = Dense(2, activation='softmax')(x)
                    custom_model = Model(inputs=resnet_50.input, outputs=predictions)

                    # Create an instance of the optimizer with the current learning rate
                    optimizer = optimizer_class(learning_rate=lr)

                    # Compile the model with current parameters
                    custom_model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])

                    # Store model name and parameters in a dictionary
                    model_name = f"ResNet50_LR{lr}_Loss{loss_function}_{optimizer_class.__name__}_Dropout{dropout_rate}_Batch{batch_size}"
                    model_names_resnet.append(model_name)

                    model_params = {
                        'model_name': model_name,
                        'learning_rate': lr,
                        'loss_function': loss_function,
                        'optimizer': optimizer_class.__name__,
                        'dropout_rate': dropout_rate,
                        'batch_size': batch_size
                    }

                    compiled_models_resnet.append({'model': custom_model, 'params': model_params})

# Print model names with corresponding parameters
for model_info in compiled_models_resnet:
    print(f"Model Name: {model_info['params']['model_name']}")
    print(f"Learning Rate: {model_info['params']['learning_rate']}")
    print(f"Loss Function: {model_info['params']['loss_function']}")
    print(f"Optimizer: {model_info['params']['optimizer']}")
    print(f"Dropout Rate: {model_info['params']['dropout_rate']}")
    print(f"Batch Size: {model_info['params']['batch_size']}")
    print("-" * 40)


# In[ ]:


# Define the train_and_evaluate_models function
def train_and_evaluate_models(X_train, y_train, X_test, y_test, compiled_models, model_names, epochs=100):
    results = []
    for model, model_name in zip(compiled_models, model_names):
        # Initialize variables for tracking maximum accuracy
        max_accuracy = 0
        max_accuracy_epoch = 0

        # Define a checkpoint to save the model when target accuracy is reached
        checkpoint = ModelCheckpoint(f'{model_name}_model_FR.h5', monitor='val_accuracy', 
                                     save_best_only=True, save_weights_only=False, mode='max', verbose=1)

        # Define early stopping to stop training if accuracy doesn't improve for 10 epochs
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)

        # Train the current model 
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), batch_size=32, callbacks=[checkpoint, early_stopping])

        current_epoch_predictions = []
        current_epoch_labels = []

        for epoch, val_accuracy in enumerate(history.history['val_accuracy'], 1):
            # Check if the current epoch achieves higher accuracy than the previous maximum
            if val_accuracy > max_accuracy:
                max_accuracy = val_accuracy
                max_accuracy_epoch = epoch
                current_epoch_predictions = model.predict(X_test)
                current_epoch_labels = y_test.argmax(axis=1)

        # Get the maximum accuracy and the corresponding epoch
        max_accuracy_epoch = np.argmax(history.history['val_accuracy'])
        max_accuracy = history.history['val_accuracy'][max_accuracy_epoch]

        # Evaluate the model on the test data
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f'Test Accuracy for {model_name}: {test_accuracy}')

        # Store results
        results.append({'Model': model_name, 'Accuracy': max_accuracy})

        print("-" * 40)  # Print a line of dashes 
        print(f"Maximum accuracy of {max_accuracy:.2f} achieved at epoch {max_accuracy_epoch+1}")
        print("Model with high accuracy is saved using the keras ModelCheckpoint")
        print("-" * 40)  # Print a line of dashes 
        print("\n")

        # Assuming model.predict returns probabilities for each class
        y_pred_probs = model.predict(X_test)

        # Convert probabilities to class labels
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Convert true labels to class labels if y_test is one-hot encoded
        y_true = np.argmax(y_test, axis=1)

        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(model_name)
        plt.show()

    return results


# In[ ]:


# Call the train_and_evaluate_models function with compiled ResNet-50 models
results_resnet = train_and_evaluate_models(X_train, y_train, X_test, y_test, [model_info['model'] for model_info in compiled_models_resnet], model_names_resnet)


# In[ ]:


import pandas as pd

# Create an empty list to store individual batch results
all_results = []

# Number of models
num_models = len(compiled_models_resnet)
batch_size = 12

# Loop through models in batches of 12
for i in range(0, num_models, batch_size):
    # Get a batch of models
    model_batch = [model_info['model'] for model_info in compiled_models_resnet[i:i+batch_size]]
    # Get corresponding model names
    model_names_batch = model_names_resnet[i:i+batch_size]
    # Call the train_and_evaluate_models function with the batch of models
    results_resnet_batch = train_and_evaluate_models(X_train, y_train, X_test, y_test, model_batch, model_names_batch)
    # Append the batch results to the list
    all_results.append(pd.DataFrame(results_resnet_batch))

# Concatenate all batch results into a single DataFrame
resnet_results = pd.concat(all_results, ignore_index=True)

# Set display option to show full content of the columns
pd.set_option('display.max_colwidth', None)

# Print the full DataFrame
print(resnet_results)


# In[70]:


import pandas as pd
import numpy as np

# Assuming all_results is a 3D list like [[[1, 2], [3, 4]], [[5, 6], [7, 8]], ...]
# Flatten the 3D list into a 2D list
flattened_list = np.array(all_results).reshape(-1, 2)

# Convert the flattened list to a DataFrame
df = pd.DataFrame(flattened_list, columns=['Model', 'Accuracy'])

# Now, 'df' is your DataFrame


# In[74]:


df


# In[75]:


import pandas as pd

# Assuming df is your DataFrame
# Sort the DataFrame by the 'Accuracy' column in descending order
sorted_df = df.sort_values(by='Accuracy', ascending=False)

# Now, 'sorted_df' contains the sorted DataFrame


# In[76]:


sorted_df


# ### The END of Code

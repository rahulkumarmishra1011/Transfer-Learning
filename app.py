import zipfile

# Download zip file of pizza_steak images
zip_ref = "https://huggingface.co/spaces/rahulmishra/model/resolve/main/10_food_classes_10_percent.zip"

# Unzip the downloaded file
zip_ref = zipfile.ZipFile("10_food_classes_10_percent.zip", "r")
zip_ref.extractall()
zip_ref.close()

train_dir = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test/"

# Get helper_functions.py script from course GitHub
# Import helper functions we're going to use
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir

# How many images/classes are there?
walk_through_dir("10_food_classes_10_percent")

import tensorflow as tf
IMG_SIZE = (224, 224)
train_data_all_10_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                                label_mode="categorical",
                                                                                image_size=IMG_SIZE)
                                                                                
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode="categorical",
                                                                image_size=IMG_SIZE,
                                                                shuffle=False)

# Import the required modules for model creation
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

## NEW: Newer versions of TensorFlow (2.10+) can use the tensorflow.keras.layers API directly for data augmentation
data_augmentation = Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
  layers.RandomHeight(0.2),
  layers.RandomWidth(0.2),
  # preprocessing.Rescaling(1./255) # keep for ResNet50V2, remove for EfficientNetB0
], name ="data_augmentation")

# Setup base model and freeze its layers (this will extract features)
base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False)
base_model.trainable = False

# Setup model architecture with trainable top layers
inputs = layers.Input(shape=(224, 224, 3), name="input_layer") # shape of input image
x = data_augmentation(inputs) # augment images (only happens during training)
x = base_model(x, training=False) # put the base model in inference mode so we can use it to extract features without updating the weights
x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x) # pool the outputs of the base model
outputs = layers.Dense(len(train_data_all_10_percent.class_names), activation="softmax", name="output_layer")(x) # same number of outputs as classes
model = tf.keras.Model(inputs, outputs)

# Get a summary of our model
model.summary()


# Compile
model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(), # use Adam with default settings
              metrics=["accuracy"])

# Fit
history_all_classes_10_percent = model.fit(train_data_all_10_percent,
                                           epochs=20, # fit for 5 epochs to keep experiments quick
                                           validation_data=test_data,
                                           validation_steps=int(0.15 * len(test_data)) # evaluate on smaller portion of test data
                        )

# Evaluate model 
results_feature_extraction_model = model.evaluate(test_data)
results_feature_extraction_model

# Unfreeze all of the layers in the base model
base_model.trainable = True

# Refreeze every layer except for the last 5
for layer in base_model.layers[:-5]:
  layer.trainable = False

# Recompile model with lower learning rate
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4), # 10x lower learning rate than default
              metrics=['accuracy'])
# What layers in the model are trainable?
for layer in model.layers:
  print(layer.name, layer.trainable)

# Fine-tune for 5 more epochs
fine_tune_epochs = 10# model has already done 5 epochs, this is the total number of epochs we're after (5+5=10)

history_all_classes_10_percent_fine_tune = model.fit(train_data_all_10_percent,
                                                     epochs=fine_tune_epochs,
                                                     validation_data=test_data,
                                                     validation_steps=int(0.15 * len(test_data)), # validate on 15% of the test data
                                                     initial_epoch=history_all_classes_10_percent.epoch[-1])

# Evaluate fine-tuned model on the whole test dataset
results_all_classes_10_percent_fine_tune = model.evaluate(test_data)
results_all_classes_10_percent_fine_tune

# Make predictions with model
pred_probs = model.predict(test_data, verbose=1) # set verbosity to see how long it will take 

# We get one prediction probability per class
print(f"Number of prediction probabilities for sample 0: {len(pred_probs[0])}")
print(f"What prediction probability sample 0 looks like:\n {pred_probs[0]}")
print(f"The class with the highest predicted probability by the model for sample 0: {pred_probs[0].argmax()}")


# Get the class predicitons of each label
pred_classes = pred_probs.argmax(axis=1)

# How do they look?
pred_classes[:10]



# Note: This might take a minute or so due to unravelling 790 batches
y_labels = []
for images, labels in test_data.unbatch(): # unbatch the test data and get images and labels
  y_labels.append(labels.numpy().argmax()) # append the index which has the largest value (labels are one-hot)
y_labels[:10] # check what they look like (unshuffled)


def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).
  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  #img = tf.io.read_file(filename)
  # Decode it into a tensor
  #img = tf.io.decode_image(img)
  # Resize the image
  img = filename
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img

class_names = test_data.class_names

# Make preds on a series of random images

import os
import random
import matplotlib.pyplot as plt

plt.figure(figsize=(17, 10))
for i in range(10):
  # Choose a random image from a random class 
  class_name = random.choice(class_names)
  filename = random.choice(os.listdir(test_dir + "/" + class_name))
  filepath = test_dir + class_name + "/" + filename

  # Load the image and make predictions
 # img = load_and_prep_image(filepath, scale=False) # don't scale images for EfficientNet predictions
  ##pred_class = class_names[pred_prob.argmax()] # find the predicted class 

  # Plot the image(s)
  #plt.subplot(1, 10, i+1)
  #plt.imshow(img/255.)
  #if class_name == pred_class: # Change the color of text based on whether prediction is right or wrong
   # title_color = "g"
  #else:
   # title_color = "r"
  #plt.title(f"actual: {class_name}, pred: {pred_class}, prob: {pred_prob.max():.2f}", c=title_color)
  #plt.axis(False);


class_name = random.choice(class_names)
filename = random.choice(os.listdir(test_dir + "/" + class_name))
filepath = test_dir + class_name + "/" + filename


def pred(filepath):
   img = load_and_prep_image(filepath, scale=False) # don't scale images for EfficientNet predictions
   pred_prob = model.predict(tf.expand_dims(img, axis=0)) # model accepts tensors of shape [None, 224, 224, 3]
   pred_class = class_names[pred_prob.argmax()] # find the predicted class 
   return pred_class


# Get accuracy score by comparing predicted classes to ground truth labels
from sklearn.metrics import accuracy_score
sklearn_accuracy = accuracy_score(y_labels, pred_classes)
sklearn_accuracy

from pathlib import Path
# Create a list of example inputs to our Gradio demo
test_data_paths = list(Path(test_dir).glob("*/*.jpg"))

example_list = [[str(filepath)] for filepath in random.sample(test_data_paths, k=10)]
example_list


import gradio as gr

# Create title, description and article strings
title = "FoodVision ADVANCE 10 food items 🥩🍕 with transfer learning"
description = "🍽️🔍 In this project, we explore the world of food 🌮🍕🥗 by building a transfer learning-based feature extractor 🧠📸. By leveraging the power of pre-trained models 🚀, we aim to classify 10 different food items with delicious accuracy 📊🍔. Let's dive into the mouth-watering world of food classification! 🍩🍓🍜"


# Create the Gradio demo
demo = gr.Interface(fn=pred, # mapping function from input to output
                    inputs=["image"], # what are the inputs?
                    outputs=["text"], # our fn has two outputs, therefore we have two outputs
                    examples=example_list, 
                    title=title,
                    description=description)

# Launch the demo!
#demo.launch(share=True,debug=True) # generate a publically shareable URL?
demo.launch(inline=True)
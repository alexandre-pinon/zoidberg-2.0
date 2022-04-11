import os
import random

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def view_random_images(target_dir, class_names):
    """
  Read and show 10 random images from the dataset
  """
    n_images = 10
    fig, ax = plt.subplots(2, 5, figsize=(24, 8))

    # Read in the image and plot it using matplotlib
    for i in range(n_images):
        target_class = random.choice(class_names)
        target_folder = target_dir + target_class

        # Get a random image path
        random_image = random.sample(os.listdir(target_folder), 1)[0]

        index = (0, i) if i < 5 else (1, i - 5)
        img = mpimg.imread(f'{target_folder}/{random_image}')
        ax[index].imshow(img, cmap='bone')
        ax[index].set_title(target_class)
        ax[index].set_frame_on(False)
        ax[index].set_xticks([])
        ax[index].set_yticks([])
        ax[index].set_xlabel(f"{img.shape}")


def plot_loss_curves(history):
    """
  Returns separate loss curves for training and validation metrics.
  """
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax[0].plot(epochs, loss, label='training_loss', c='r')
    ax[0].plot(epochs, val_loss, label='val_loss', c='orange')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()

    # Plot accuracy
    ax[1].plot(epochs, accuracy, label='training_accuracy', c='g')
    ax[1].plot(epochs, val_accuracy, label='val_accuracy', c='lightgreen')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()


def evaluate_model_binary(model, test_data):
    """
  Evaluate BINARY model on test data using different metrics
  """
    y_pred = model.predict(test_data)
    y_pred = np.round(y_pred.flatten())
    print(f'Accuracy: {accuracy_score(test_data.labels, y_pred):.1%}')
    print(f"Precision: {precision_score(test_data.labels, y_pred):.1%}")
    print(f"Recall: {recall_score(test_data.labels, y_pred):.1%}")
    print(f"F1: {f1_score(test_data.labels, y_pred):.1%}")


def make_val_predictions(model,
                         val_dir,
                         class_names,
                         img_size=(224, 224),
                         batch_size=32):
    """
  Make predictions on BINARY plot_data (obtained using flow_from_directory()) and
  graphically compare them with true labels
  """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Import the target images and preprocess them
    plot_datagen = ImageDataGenerator(rescale=1 / 255.)
    plot_data = plot_datagen.flow_from_directory(directory=val_dir,
                                                 target_size=img_size,
                                                 class_mode='binary',
                                                 batch_size=batch_size,
                                                 shuffle=False)
    # Make predictions
    preds = model.predict(plot_data).flatten()

    # Get predictions labels & percentages
    percentages = []
    preds_labels = []
    true_labels = class_names[plot_data.labels]

    for pred in preds:
        percentages.append(pred if pred >= 0.5 else 1 - pred)
        preds_labels.append(class_names[int(tf.round(pred))])

    img_paths = plot_data.filepaths
    n_images = len(preds)
    n_rows = n_cols = (n_images // 4)
    fig, axes = plt.subplots(n_rows,
                             n_cols,
                             figsize=(4 * n_cols, 3 * n_rows),
                             gridspec_kw={
                                 'wspace': 0,
                                 'hspace': .6
                             })
    axes = np.ravel(axes)

    for i, ax in enumerate(axes):
        if i == n_images:
            break

        img = mpimg.imread(img_paths[i])
        color = 'green' if preds_labels[i] == true_labels[i] else 'red'

        ax.imshow(img, cmap='bone')
        ax.set_title(
            f"Pred: {preds_labels[i]} {percentages[i]:.1%}\n(True: {true_labels[i]})",
            c=color)
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_confusion_matrix(cm, class_names):
    """
  Plot confusion matrixes:
  - The first one with absolute values
  - The second one with percentages
  """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    sns.heatmap(data=cm,
                ax=ax[0],
                cmap='RdYlGn',
                annot=True,
                fmt='.0f',
                annot_kws={'size': 16})
    ax[0].set_title('Confusion Matrix', fontsize=22)
    ax[0].set_xlabel('Predicted Label', fontsize=16)
    ax[0].set_ylabel('True Label', fontsize=16)
    ax[0].set_xticks(ticks=np.arange(.5, len(class_names) + .5))
    ax[0].set_xticklabels(labels=class_names, fontsize=12)
    ax[0].set_yticks(ticks=np.arange(.5, len(class_names) + .5))
    ax[0].set_yticklabels(labels=class_names, fontsize=12)

    sns.heatmap(data=cm / np.sum(cm, axis=1, keepdims=True),
                ax=ax[1],
                cmap='RdYlGn',
                annot=True,
                fmt='.1%',
                annot_kws={'size': 16})
    ax[1].set_title('Confusion Matrix', fontsize=22)
    ax[1].set_xlabel('Predicted Label', fontsize=16)
    ax[1].set_ylabel('True Label', fontsize=16)
    ax[1].set_xticks(ticks=np.arange(.5, len(class_names) + .5))
    ax[1].set_xticklabels(labels=class_names, fontsize=12)
    ax[1].set_yticks(ticks=np.arange(.5, len(class_names) + .5))
    ax[1].set_yticklabels(labels=class_names, fontsize=12)


def evaluate_model_multi(model, test_data, class_names):
    """
  Evaluate MULTI model on test data using different metrics
  Also plots the confusion matrixes
  Compatible with DirectoryIterator as test_data
  """
    y_pred = model.predict(test_data)
    y_pred = y_pred.argmax(axis=1)
    print(f'Accuracy: {accuracy_score(test_data.labels, y_pred):.1%}')
    print(
        f"Precision: {precision_score(test_data.labels, y_pred, average='weighted'):.1%}"
    )
    print(
        f"Recall: {recall_score(test_data.labels, y_pred, average='weighted'):.1%}"
    )
    print(f"F1: {f1_score(test_data.labels, y_pred, average='weighted'):.1%}")

    cm = confusion_matrix(test_data.labels, y_pred)
    plot_confusion_matrix(cm, class_names)


def evaluate_model_multi_batchdataset(model, test_data, class_names):
    """
  Evaluate MULTI model on test data using different metrics
  Also plots the confusion matrixes
  Compatible with BatchDataset as test_data
  """
    y_pred = model.predict(test_data)
    y_pred = y_pred.argmax(axis=1)
    labels = np.concatenate([np.argmax(y.numpy(), axis=-1) for _, y in test_data])

    print(f'Accuracy: {accuracy_score(labels, y_pred):.1%}')
    print(
        f"Precision: {precision_score(labels, y_pred, average='weighted'):.1%}"
    )
    print(f"Recall: {recall_score(labels, y_pred, average='weighted'):.1%}")
    print(f"F1: {f1_score(labels, y_pred, average='weighted'):.1%}")

    cm = confusion_matrix(labels, y_pred)
    plot_confusion_matrix(cm, class_names)


def create_tensorboard_callback(dir_name, experiment_name):
    """
  Creates a tensorboard callback given a directory and an experiment name
  """
    import datetime
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now(
    ).strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


def create_tfhub_model(model_url, num_classes=10, img_size=(224, 224)):
    """Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.
  
  Args:
    model_url (str): A TensorFlow Hub feature extraction URL.
    num_classes (int): Number of output neurons in output layer,
      should be equal to number of target classes, default 10.

  Returns:
    An uncompiled Keras Sequential model with model_url as feature
    extractor layer and Dense output layer with num_classes outputs.
  """
    import tensorflow_hub as hub
    from tensorflow.keras import layers

    # Download the pretrained model and save it as a Keras layer
    feature_extractor_layer = hub.KerasLayer(
        model_url,
        trainable=False,  # freeze the underlying patterns
        name='feature_extraction_layer',
        input_shape=img_size + (3, ))  # define the input image shape

    # Create our own model
    model = tf.keras.Sequential([
        feature_extractor_layer,  # use the feature extraction layer as the base
        layers.Dense(num_classes, activation='softmax',
                     name='output_layer')  # create our own output layer      
    ])

    return model


def walk_through_dir(dir_path):
    """
  Walks through dir_path returning its contents.
  Args:
    dir_path (str): target directory
  
  Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
  """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(
            f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'."
        )

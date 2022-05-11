from argparse import ArgumentError
import os
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


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


def preprocess_images(ds,
                      seed,
                      shuffle=False,
                      augment=False,
                      img_size=(224, 224),
                      batch_size=32,
                      shuffle_buffer_size=1000):
    """
    Preprocess a tf.data.Dataset of images using rescaling and various data augmentation techinques
    """

    def resize_and_rescale(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [*img_size])
        image = (image / 255.0)
        return image, label

    def augment(image_label, seed):
        image, label = image_label
        image, label = resize_and_rescale(image, label)
        image = tf.image.resize_with_crop_or_pad(image, img_size[0] + 6,
                                                 img_size[1] + 6)

        # Stateless image augmentation
        new_seeds = tf.random.experimental.stateless_split(seed, num=7)
        image = tf.image.stateless_random_crop(image,
                                               size=[*img_size, 3],
                                               seed=seed)
        image = tf.image.stateless_random_brightness(image,
                                                     max_delta=0.2,
                                                     seed=new_seeds[0, :])
        image = tf.image.stateless_random_contrast(image,
                                                   lower=0.8,
                                                   upper=1.2,
                                                   seed=new_seeds[1, :])
        image = tf.image.stateless_random_flip_left_right(image,
                                                          seed=new_seeds[2, :])
        image = tf.image.stateless_random_flip_up_down(image,
                                                       seed=new_seeds[3, :])
        image = tf.image.stateless_random_hue(image,
                                              max_delta=0.2,
                                              seed=new_seeds[4, :])
        image = tf.image.stateless_random_saturation(image,
                                                     lower=0.8,
                                                     upper=1.2,
                                                     seed=new_seeds[6, :])

        image = tf.clip_by_value(image, 0, 1)
        return image, label

    # Create a wrapper function for updating seeds.
    def augment_data(x, y):
        seed = rng.make_seeds(2)[0]
        image, label = augment((x, y), seed)
        return image, label

    def keras_augment_data(x, y):
        keras_augment = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomTranslation(0.2, 0.2),
            tf.keras.layers.RandomZoom(0.2, 0.2)
        ])
        return keras_augment(x, training=True), y

    ds = ds.unbatch()

    if shuffle:
        ds = ds.shuffle(shuffle_buffer_size, seed)

    # Use data augmentation only on the training set.
    if augment:
        # Create a generator.
        rng = tf.random.Generator.from_seed(seed, alg='philox')
        ds = ds.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(keras_augment_data, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(resize_and_rescale, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=tf.data.AUTOTUNE)


def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.
    """

    epochs = range(len(history.history['loss']))

    fig, ax = plt.subplots(1, 4, figsize=(30, 5))

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    ax[1].plot(epochs, accuracy, label='training_accuracy', c='g')
    ax[1].plot(epochs, val_accuracy, label='val_accuracy', c='lightgreen')
    ax[1].set_title('Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()

    precision = history.history['precision']
    val_precision = history.history['val_precision']
    ax[2].plot(epochs, precision, label='training_precision', c='darkblue')
    ax[2].plot(epochs, val_precision, label='val_precision', c='royalblue')
    ax[2].set_title('Precision')
    ax[2].set_xlabel('Epochs')
    ax[2].legend()

    recall = history.history['recall']
    val_recall = history.history['val_recall']
    ax[3].plot(epochs, recall, label='training_recall', c='darkcyan')
    ax[3].plot(epochs, val_recall, label='val_recall', c='c')
    ax[3].set_title('Recall')
    ax[3].set_xlabel('Epochs')
    ax[3].legend()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    ax[0].plot(epochs, loss, label='training_loss', c='r')
    ax[0].plot(epochs, val_loss, label='val_loss', c='orange')
    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()


def calculate_metrics(y_true, y_pred, multi=False):
    average = 'weighted' if multi else 'binary'
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "f1": f1_score(y_true, y_pred, average=average)
    }


def evaluate_model_binary(model, test_data):
    """
    Evaluate BINARY model on test data using different metrics
    Compatible with DirectoryIterator as test_data
    """
    y_pred = model.predict(test_data)
    y_pred = np.round(y_pred.flatten())

    y_true = test_data.labels

    metrics = calculate_metrics(y_true, y_pred)

    return y_true, y_pred, metrics


def evaluate_model_binary_batchdataset(model, test_data):
    """
    Evaluate BINARY model on test data using different metrics
    Compatible with BatchDataset as test_data
    """
    y_pred = model.predict(test_data)
    y_pred = np.round(y_pred.flatten())
    labels = np.concatenate([np.ravel(y.numpy()) for _, y in test_data])

    y_true = labels

    metrics = calculate_metrics(y_true, y_pred)

    return y_true, y_pred, metrics


def evaluate_model_multi(model, test_data):
    """
    Evaluate MULTI model on test data using different metrics
    Also plots the confusion matrixes
    Compatible with DirectoryIterator as test_data
    """
    y_pred = model.predict(test_data)
    y_pred = y_pred.argmax(axis=1)

    y_true = test_data.labels

    metrics = calculate_metrics(y_true, y_pred, multi=True)

    return y_true, y_pred, metrics


def evaluate_model_multi_batchdataset(model, test_data):
    """
    Evaluate MULTI model on test data using different metrics
    Also plots the confusion matrixes
    Compatible with BatchDataset as test_data
    """
    y_pred = model.predict(test_data)
    y_pred = y_pred.argmax(axis=1)
    labels = np.concatenate(
        [np.argmax(y.numpy(), axis=-1) for _, y in test_data])

    y_true = labels

    metrics = calculate_metrics(y_true, y_pred, multi=True)

    return y_true, y_pred, metrics


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


def plot_confusion_matrixes(cm, class_names, model_name=None, cmap='RdYlGn'):
    """
    Plot confusion matrixes:
    - The first one with absolute values
    - The second one with percentages
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    if model_name:
        fig.suptitle(model_name, fontsize=22)

    sns.heatmap(data=cm,
                ax=ax[0],
                cmap=cmap,
                annot=True,
                fmt='.0f',
                annot_kws={'size': 16})
    ax[0].set_xlabel('True Label', fontsize=16)
    ax[0].set_ylabel('Predicted Label', fontsize=16)
    ax[0].set_xticks(ticks=np.arange(.5, len(class_names) + .5))
    ax[0].set_xticklabels(labels=class_names, fontsize=12)
    ax[0].set_yticks(ticks=np.arange(.5, len(class_names) + .5))
    ax[0].set_yticklabels(labels=class_names, fontsize=12)

    sns.heatmap(data=cm / np.sum(cm, axis=0, keepdims=True),
                ax=ax[1],
                cmap=cmap,
                annot=True,
                fmt='.1%',
                annot_kws={'size': 16})
    ax[1].set_xlabel('True Label', fontsize=16)
    ax[1].set_ylabel('Predicted Label', fontsize=16)
    ax[1].set_xticks(ticks=np.arange(.5, len(class_names) + .5))
    ax[1].set_xticklabels(labels=class_names, fontsize=12)
    ax[1].set_yticks(ticks=np.arange(.5, len(class_names) + .5))
    ax[1].set_yticklabels(labels=class_names, fontsize=12)


def compare_models(models, test_data, type='binary'):
    if type == 'binary':
        evaluation_scores = [
            evaluate_model_binary(model, test_data) for model in models
        ]
    elif type == 'multi':
        evaluation_scores = [
            evaluate_model_multi(model, test_data) for model in models
        ]

    metrics = [score[2] for score in evaluation_scores]

    df = pd.DataFrame(metrics)
    df['model'] = [m.name for m in models]
    df.set_index('model', inplace=True)

    return evaluation_scores, df.applymap('{:.2%}'.format)


def train_new_densenet201_model(train_data,
                                val_data,
                                type='binary',
                                seed=2,
                                name=None,
                                fine_tune_model=False,
                                img_size=(224, 224),
                                epochs=20,
                                class_weight=None,
                                verbose=False):
    """
    Creates a new densenet201 model with given parameters and train it.
    Returns the trained model and the training history
    """
    tf.random.set_seed(seed)

    # Model creation
    base_model = tf.keras.applications.DenseNet201(include_top=False)
    if not fine_tune_model:
        base_model.trainable = False
        learning_rate = 1e-3
    else:
        base_model.trainable = True
        learning_rate = 1e-4
        # Unfreeze the 20 last layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False

    inputs = tf.keras.layers.Input(shape=(*img_size, 3), name="input_layer")
    x = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D(
        name="global_average_pooling_layer")(x)

    if type == 'binary':
        loss = 'binary_crossentropy'
        output_units = 1
        activation = 'sigmoid'
    elif type == 'categorical':
        loss = 'categorical_crossentropy'
        output_units = 3
        activation = 'softmax'
    else:
        raise ArgumentError(f'Invalid type {type}')

    outputs = tf.keras.layers.Dense(units=output_units,
                                    activation=activation,
                                    name="output_layer")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)

    # Model compilation
    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['accuracy', 'Precision', 'Recall'])

    # Model training
    history = model.fit(train_data,
                        epochs=epochs,
                        validation_data=val_data,
                        class_weight=class_weight,
                        verbose=verbose)

    return model, history


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

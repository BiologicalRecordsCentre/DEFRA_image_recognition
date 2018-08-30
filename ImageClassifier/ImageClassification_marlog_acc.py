# Imports for moving/copying images from repo to model use
from os import listdir, makedirs, walk
from os.path import exists, isfile, join, isdir
import shutil

import logging

# Used to save the fit() history function
import pickle

# Basic data handling imports
import numpy as np
import math
import pandas as pd
from collections import Counter

# Plotting imports and function call for plotting to notebook output
#from bokeh.charts import Histogram
# bokeh.charts is no longer maintained, removing this line and references to Histogram on line 339
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import row
from bokeh.models import HoverTool, ColumnDataSource, BasicTicker, ColorBar, ColumnDataSource, LinearColorMapper, PrintfTickFormatter
from bokeh.palettes import PRGn11, Colorblind8
from bokeh import palettes
from bokeh.transform import factor_cmap
from bokeh.transform import transform
import itertools

from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Keras image manipulation and label manipulation
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical

# Keras model and layers
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model

# Keras training callbacks
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

# Keras pre-trained model for fine-tuning
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.imagenet_utils import preprocess_input


class ImageClassification(object):
    """Image classification object, to control test/data splits, bottleneck-feature creation, top-model training,
    top-model evaluation, pre-trained model fine-tuning and evaluation.
    """

    def __init__(self,
                 instance_id,
                 image_df,
                 predict_df,
                 output_directory,
                 batch_size,
                 train_test_split,
                 train_multiplier,
                 use_weights=True,
                 using_notebook=True):
        """Initialize the image classifier.

        Arguments:
            instance_id: used to identify the instances output in the output_directory, to allow referencing previous
                runs.
            image_df: a Pandas dataframe with three columns: directory, image_id and class.  The directory is where the
                file called image_id is found, and image_id must be unique within the dataframe, not just
                per-directory.
            predict_df: a Pandas dataframe, with the same format as image_df, and must have only the same classes, the
                contents will be used for evaluating the classifier's performance after training/validation, as a
                completely different test set.
            output_directory: the base folder location to put all outputs generated into, and if the models and
                bottleneck features have been created previously in this location, they will be re-used, no previous
                 runs are ever overwritten - either the instance's directory must be removed, or the individual
                 directories for a particular model, or rather just create a new instance.
            batch_size: the batch size to use at all stages of the instance's training/data-use
            train_test_split: the percentage of each class to use for testing/validation, e.g. 0.25 for 25%
            train_multiplier: the number of times the full set of training images will be pushed through the system,
                using augmented versions from a data generator.
            use_weights: a boolean flag to indicate if class imbalance should be taken into account when training and
                to weight classes with less images as a greater loss.
            using_notebook: a boolean flag to indicate if this instance is being used within a Jupyter Notebook, as the
                bokeh plotting has a setting for that environment.
        """

        # Set instance variables from constructor parameters
        self.instance_id = instance_id
        self.image_df = image_df
        self.predict_dataset = predict_df
        self.output_directory = output_directory
        self.batch_size = batch_size
        self.train_test_split = train_test_split
        self.train_multiplier = train_multiplier
        self.use_weights = use_weights
        self.using_notebook = using_notebook

        # Create derived variables
        self.class_counts = self.__get_class_counts()
        self.class_labels = self.__get_class_labels()
        self.class_weights = self.__get_class_weights()

        # The model list, used after training to list all the trained models available for analysis/prediction
        self.model_list = []

        # Add one-hot-encoded label column to the image dataframe used for train/test purposes
        self.image_df['one-hot-label'] = None
        for idx, row in self.image_df.iterrows():
            self.image_df.loc[idx, 'one-hot-label'] = self.class_labels[row['class']]

        # Create a list of available models, to make sure that those specified by the user
        #  are supported
        self.available_models = ['xception', 'vgg16', 'resnet50', 'inceptionv3', 'inception-resnet']

        # Create a list of TF based models, as these require a different image normalization function to the other types
        self.tf_models = ['xception', 'inceptionv3', 'inception-resnet']

        # The height/width dimension value of each model, currently set to the default size for each model, but this
        #  could be changed to larger or smaller images depending on performance.
        self.model_image_dimensions = {'xception': 299,
                                       'vgg16': 224,
                                       'resnet50': 224,
                                       'inceptionv3': 299,
                                       'inception-resnet': 299}

        # The feature size output from the pre-trained models before the topmost softmax classifier.  In the case of
        #  xception and inception this is a single value due to performing global average 2d pooling at the last step.
        # Resnet50 also has the same pooling technique, but keeps the three dimension representation.
        self.model_dense_pooled_dimensions = {'xception': (2048,),
                                       'vgg16': (7, 7, 512),
                                       'resnet50': (1, 1, 2048),
                                       'inceptionv3': (2048,),
                                       'inception-resnet': (1536,)}

        # Set the Bokeh notebook output state if using a Jupyter Notebook
        if using_notebook:
            output_notebook()

        # Set the output directory locations, these are used to group outputs by instance and type of output, e.g.
        #  bottleneck features, or trained model parameters.
        self.base_instance_directory = join(self.output_directory, self.instance_id)
        self.split_image_directory = join(self.output_directory, self.instance_id, "split-data")
        self.bottleneck_directory = join(self.output_directory, "bottleneck-data")
        self.model_directory = join(self.output_directory, self.instance_id, "trained-models")

        # Create the main output and base instance directories if they don't exist, as functions within this class
        #  expect them to be created
        if not exists(join(self.output_directory)):
            makedirs(join(self.output_directory))
        if not exists(join(self.base_instance_directory)):
            makedirs(join(self.base_instance_directory))

        # Create or load the train / test split of the self.image_df data frame, depending if they have been created
        #  previously or not.
        self.train_data, self.test_data = self.__create_train_test_groups()

        # Add one-hot-encoded label column to the image dataframe for prediction, final testing purposes
        self.predict_dataset['one-hot-label'] = None
        for idx, row in self.predict_dataset.iterrows():
            self.predict_dataset.loc[idx, 'one-hot-label'] = self.class_labels[row['class']]

        # Check that all of the rows in the prediction data frame have been set, if any have not been then this means
        #  there are classes in the prediction data that are not in the train/test data.
        if (pd.isnull(self.predict_dataset['one-hot-label'])).any():
            raise Exception("Classes in the predict dataset are not present in the training dataset, cannot progress.")

        # Create the data frames that will hold the predicted classes
        self.predicted_dataset = None
        self.predicted_class_probabilities = None

    # =================================================================================================================
    # Helper Functions: data summary functions, label encoding, class balance weight creation
    # =================================================================================================================

    def __get_class_counts(self):
        """Create a data frame with the total number of example images per class."""
        label_count = (self.image_df.groupby(['class'], as_index=False)
                       .count()
                       .sort_values('image_id', ascending=False)
                       .loc[:, ('class', 'image_id')])
        label_count.columns = ["class", "number_instances"]
        return label_count

    def __get_class_labels(self):
        """Convert the alphanumeric class labels to one-hot-encoded representation, and return a dictionary mapping the
        textual class to the one-hot-encoded class.
        """
        class_label_dict = {}
        sorted_classes = (self.image_df
                          .drop_duplicates('class')
                          .loc[:, 'class']
                          .sort_values(ascending=True, inplace=False)
                          .reset_index(drop=True))
        categorical_classes = to_categorical(sorted_classes.index.values.tolist())
        for curr_pos in range(len(sorted_classes)):
            class_label_dict[sorted_classes[curr_pos]] = categorical_classes[curr_pos]

        return class_label_dict

    def __get_class_weights(self):
        """Computes the weighting of each class compared to the class with the maximum amount of samples, and returns a
        dictionary of these values.  The dictionary key is the class numeric position in the one-hot-encoding scheme.
        """
        class_weight_dict = {}
        class_counts = Counter(self.image_df['class'].tolist())
        max_count = class_counts.most_common(1)[0][1]

        sorted_classes = (self.image_df
                          .drop_duplicates('class')
                          .loc[:, 'class']
                          .sort_values(ascending=True, inplace=False)
                          .reset_index(drop=True))

        for idx, value in sorted_classes.iteritems():
            class_weight_dict[idx] = max_count / class_counts[value]

        return class_weight_dict

    def __get_trained_models(self):
        """Searches for trained models and provides a list of the model id's to be used in evaluation of the training
        and prediction capabilities of those models.
        """

        model_directories = [current_directory
                             for current_directory in listdir(self.model_directory)
                             if isdir(join(self.model_directory, current_directory))]

        # For each model directory, check for a checkpoint file and a history file, return only those directories
        #  with both of these
        trained_model_directories = [current_directory
                                     for current_directory in model_directories
                                     if 'best-checkpoint.h5' in listdir(join(self.model_directory, current_directory))
                                     and 'history.pkl' in listdir(join(self.model_directory, current_directory))]
        return trained_model_directories

    # =================================================================================================================
    # EDA Summary: Show the classes with the most/least example images
    # =================================================================================================================

    def eda_summary(self):
        self.__print_head_tail_classes(10)
        self.__print_class_distribution()
        # The below takes too long for anything more than trivial amounts
        #self.__print_image_size_counts()

    def __print_head_tail_classes(self, num_classes):
        """Print two plots showing the most and least numerous classes, up to num_classes.
        Argument:
            num_classes: an integer specifying the number of classes to display from the head / tail of the dataframe
        """
        if self.class_counts.shape[0] > 0:
            if self.class_counts.shape[0] < num_classes:
                num_classes = self.class_counts.shape[0]

            x_axis_max = self.class_counts.head(1).reset_index(drop=True).loc[0, 'number_instances']
            most_class_figure = figure(y_range=self.class_counts['class'].head(num_classes)[::-1].tolist(),
                                       x_range=(0, x_axis_max),
                                       title='Classes with Most Examples',
                                       plot_width=475,
                                       toolbar_location=None)

            most_class_figure.hbar(y=self.class_counts['class'].head(num_classes)[::-1].tolist(),
                                   right=self.class_counts['number_instances'].head(num_classes)[::-1].tolist(),
                                   height=0.7)

            most_class_figure.xaxis.axis_label = "Class"
            most_class_figure.yaxis.axis_label = "Number of Training Image Examples"

            x_axis_max = (self.class_counts
                          .tail(num_classes)
                          .reset_index(drop=True)
                          .iloc[-num_classes]['number_instances'])
            least_class_figure = figure(y_range=self.class_counts['class'].tail(num_classes)[::-1].tolist(),
                                        x_range=(0, x_axis_max),
                                        title='Classes with Least Examples',
                                        plot_width=475,
                                        toolbar_location=None)

            least_class_figure.hbar(y=self.class_counts['class'].tail(num_classes)[::-1].tolist(),
                                    right=self.class_counts['number_instances'].tail(num_classes)[::-1].tolist(),
                                    height=0.7)

            least_class_figure.xaxis.axis_label = "Class"
            least_class_figure.yaxis.axis_label = "Number of Training Image Examples"
            show(row(most_class_figure, least_class_figure))
        else:
            logging.warning("Not enough classes to produce a plot")

    def __print_class_distribution(self):
        """Print the number of images per class as a bar-chart to view frequency, and a summarized histogram of the
        same data.
        """
        if self.class_counts.shape[0] > 2:
            mean_count = round(self.class_counts.mean(numeric_only=True).number_instances)
            median_count = round(self.class_counts.median(numeric_only=True).number_instances)

            local_counts = self.class_counts
            local_counts['Position'] = 'Greater than mean'
            local_counts.loc[(local_counts.number_instances == mean_count), 'Position'] = 'Mean'
            local_counts.loc[(local_counts.number_instances < mean_count), 'Position'] = 'Lesser than mean'
            local_counts.loc[(local_counts.number_instances == median_count), 'Position'] = 'Median'

            label_count_dict = local_counts.to_dict('list')
            label_count_dict['index'] = list(range(0, len(label_count_dict['number_instances'])))
            factors = local_counts.Position.unique().tolist()
            source = ColumnDataSource(label_count_dict)

            all_classes = figure(x_range=(0, local_counts.shape[0] - 1),
                                 title='Class Instance Amount Shape',
                                 toolbar_location=None, tools="", plot_width=475)

            all_classes.vbar(x='index',
                             source=source,
                             top='number_instances',
                             width=0.9,
                             legend='Position',
                             fill_color=factor_cmap('Position', palette=Colorblind8, factors=factors))

            all_classes.legend.location = 'top_right'
            all_classes.xaxis.axis_label = "Dataframe Index of Class instance counts"
            all_classes.yaxis.axis_label = "Number of Training Image Examples"

            # Create a histogram of occurrences
            #density_plot = Histogram(local_counts.number_instances,
            #                         title="Class Occurrence Density",
            #                         toolbar_location=None,
            #                         plot_width=475)

            #show(row(all_classes, density_plot))
        else:
            logging.warning("Not enough classes to produce a plot")

    def __print_image_size_counts(self):
        """Iterate over all images in the training and validation datasets and display the image sizes"""

        img_sizes = []
        for idx, row in self.image_df.iterrows():
            img = load_img(join(row['directory'], row['image_id']))
            img_sizes.append(img_to_array(img).shape[0:2])

        for idx, row in self.predict_dataset.iterrows():
            img = load_img(join(row['directory'], row['image_id']))
            img_sizes.append(img_to_array(img).shape[0:2])

        img_count = Counter(img_sizes)
        x_values = []
        y_values = []
        radii = []

        for current_key in img_count.keys():
            radii.append(img_count.get(current_key)*2)
            x_values.append(current_key[0])
            y_values.append(current_key[1])

        p = figure(x_range=(0, max(x_values)), y_range=(0, max(y_values)), tools="")
        p.title.text = "Image Dimensions by Count"
        p.xaxis.axis_label = "Image Width"
        p.yaxis.axis_label = "Image Height"
        p.circle(x_values, y_values, radius=radii, fill_alpha=0.3, line_color=None, fill_color='green')
        show(p)

    # =================================================================================================================
    # Train - Test data split: Take the class images and split into train/test groups, then create directory structure.
    # =================================================================================================================

    def __create_train_test_groups(self):
        """Iterate over the classes in the image dataframe and create both train and test set dataframes."""
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()

        if (exists(join(self.base_instance_directory, "train_images.csv"))
            and exists(join(self.base_instance_directory, "test_images.csv"))):
                train_df = pd.read_csv(join(self.base_instance_directory, "train_images.csv"))
                test_df = pd.read_csv(join(self.base_instance_directory, "test_images.csv"))
        else:
            for _, class_row in self.class_counts.iterrows():
                num_images = class_row['number_instances']
                num_test_images = round(self.train_test_split * num_images)

                class_idx_positions = self.image_df.loc[(self.image_df['class'] == class_row['class']), :].index.values.tolist()
                train_ids = np.random.choice(class_idx_positions, size=num_images-num_test_images, replace=False)
                test_ids = [index for index in class_idx_positions if index not in train_ids]

                train_df = train_df.append(self.image_df.loc[train_ids, :], ignore_index=True)
                test_df = test_df.append(self.image_df.loc[test_ids, :], ignore_index=True)

            train_df.to_csv(join(self.base_instance_directory, "train_images.csv"))
            test_df.to_csv(join(self.base_instance_directory, "test_images.csv"))

        return train_df, test_df

    def __create_directory_train_test_split(self):
        """With a data frame each for train and test data, create the class image directories for train and test data.
        If these already exist do nothing."""

        if (exists(join(self.split_image_directory, "train"))
            and exists(join(self.split_image_directory, "test"))):
            logging.warning("Using previously created train test class split")
        else:
            # Delete the training and testing target output directories if they already exist
            if exists(join(self.split_image_directory, "train")):
                shutil.rmtree(join(self.split_image_directory, "train"))

            if exists(join(self.split_image_directory, "test")):
                shutil.rmtree(join(self.split_image_directory, "test"))

            # Create the target and test output directories
            makedirs(join(self.split_image_directory, "train"))
            makedirs(join(self.split_image_directory, "test"))

            # Create the class sub-folders
            for current_class in self.train_data.loc[:, 'class'].unique().tolist():
                makedirs(join(self.split_image_directory, "train", current_class))
                makedirs(join(self.split_image_directory, "test", current_class))

            # Iterate through the images, and copy to the corresponding class directories
            for _, row in self.train_data.iterrows():
                file = join(row['directory'], row['image_id'])
                target_file = join(self.split_image_directory, "train", row['class'], row['image_id'])
                shutil.copyfile(file, target_file)

            # Iterate through the images, and copy to the corresponding class directories
            for _, row in self.test_data.iterrows():
                file = join(row['directory'], row['image_id'])
                target_file = join(self.split_image_directory, "test", row['class'], row['image_id'])
                shutil.copyfile(file, target_file)

    # =================================================================================================================
    # Top Model Training - Feature Creation: use the train/test split data to create pre-made model features from the
    #  convolution layers.
    # =================================================================================================================

    @staticmethod
    def tf_image_pre_processing(x):
        """Used to pre-process images being used with TensorFlow based models.

        Arguments:
            x: A single image past-in through the Keras flow form directory generators.
        """
        x /= 255.
        x -= 0.5
        x *= 2.
        return x

    def __create_bottleneck_features(self, model_type_list):
        """Using the directories with train and test data, create bottleneck features of all of the models being used
        with this instance.
        """

        # Bottleneck feature file ID prefix
        bottleneck_prefix = "bottleneck_features_"
        bottleneck_label_prefix = "bottleneck_feature_labels_"

        # For each model provided, create the bottleneck features
        for current_model in model_type_list:
            # Create different pre-processing generators depending on the model type
            if current_model in self.tf_models:
                # Create the generator for the test data
                train_generator = ImageDataGenerator(rotation_range=40,
                                                     width_shift_range=0.2,
                                                     height_shift_range=0.2,
                                                     shear_range=0.2,
                                                     zoom_range=0.2,
                                                     horizontal_flip=True,
                                                     fill_mode='nearest',
                                                     preprocessing_function=self.tf_image_pre_processing)

                # Create the generator for the validation data
                test_generator = ImageDataGenerator(preprocessing_function=self.tf_image_pre_processing)
            else:
                # Create the generator for the test data
                train_generator = ImageDataGenerator(rotation_range=40,
                                                     width_shift_range=0.2,
                                                     height_shift_range=0.2,
                                                     rescale=1. / 255,
                                                     shear_range=0.2,
                                                     zoom_range=0.2,
                                                     horizontal_flip=True,
                                                     fill_mode='nearest')

                # Create the generator for the validation data
                test_generator = ImageDataGenerator(rescale=1. / 255)

            img_dim = self.model_image_dimensions.get(current_model)
            model = self.get_model(current_model)

            train_batches = round(
                    ((self.class_counts['number_instances'].sum() * (1 - self.train_test_split)) / self.batch_size)
                    * self.train_multiplier)

            test_batches = round(
                    ((self.class_counts['number_instances'].sum() * self.train_test_split) / self.batch_size))

            self.__directory_bottleneck_feature_writing(model,
                                                      current_model,
                                                      img_dim,
                                                      'train',
                                                      train_generator,
                                                      train_batches,
                                                      bottleneck_prefix,
                                                      bottleneck_label_prefix)

            self.__directory_bottleneck_feature_writing(model,
                                                      current_model,
                                                      img_dim,
                                                      'test',
                                                      test_generator,
                                                      test_batches,
                                                      bottleneck_prefix,
                                                      bottleneck_label_prefix)

    def __directory_bottleneck_feature_writing(self,
                                             model,
                                             model_id,
                                             img_dim,
                                             top_folder,
                                             image_generator,
                                             number_batches,
                                             bottleneck_prefix,
                                             bottleneck_label_prefix):
        """Writes the output of the flow_from_directory data generator to disk, saving into the bottleneck directory."""

        # Create the counter to stop iteration once enough data has been created
        batch_counter = 0

        if not exists(join(self.bottleneck_directory, model_id + "_" + top_folder)):
            print("Beginning to create the {} bottleneck features for model: {}".format(top_folder, model_id))
            makedirs(join(self.bottleneck_directory, model_id + "_" + top_folder))

            for x_batch, y_batch in image_generator.flow_from_directory(join(self.split_image_directory, top_folder),
                                                                        target_size=(img_dim, img_dim),
                                                                        class_mode='categorical',
                                                                        batch_size=self.batch_size,
                                                                        shuffle=True):

                bottleneck_features = model.predict(x_batch)

                feature_file_name = bottleneck_prefix + str(batch_counter) + ".npy"
                label_file_name = bottleneck_label_prefix + str(batch_counter) + ".npy"

                np.save(open(join(self.bottleneck_directory, model_id + "_" + top_folder, feature_file_name), 'wb'), bottleneck_features)
                np.save(open(join(self.bottleneck_directory, model_id + "_" + top_folder, label_file_name), 'wb'), y_batch)

                batch_counter += 1
                if batch_counter > number_batches:
                    break
            del model
        else:
            logging.warning("Using previously created bottleneck features for model: {}.".format(model_id))

    # =================================================================================================================
    # Top Model Training
    # =================================================================================================================

    @staticmethod
    def get_model(model):
        """Using the string ID of the model, create an instance of a pre-trained CovNet and return."""
        if model == 'xception':
            return Xception(weights='imagenet', include_top=False, pooling='avg', input_shape=(299, 299, 3))
        elif model == 'vgg16':
            return VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif model == 'resnet50':
            return ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        elif model == 'inceptionv3':
            return InceptionV3(weights='imagenet', include_top=False, pooling='avg', input_shape=(299, 299, 3))
        elif model == 'inception-resnet':
            return InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(299, 299, 3))
        else:
            raise Exception("No matching model selected.")

    def __bottleneck_data_generator(self, data_directory):
        """Bottleneck generator iterates over the previously created bottleneck feature files found in the directory
        and yields them as training or validation samples.  The files must be bottleneck features, from the last layer
        in the pre-trained model before the classification layer.

        As the generator is controlled by the training function, it loops once all features have been exhausted.

        Arguments:
            data_directory: the location of numpy files, different for every model.
        """
        while True:
            # Read the files of features and labels
            data_file_list = [file for file in listdir(data_directory)
                              if isfile(join(data_directory, file)) and "labels" not in file]

            label_file_list = [file for file in listdir(data_directory)
                               if isfile(join(data_directory, file)) and "labels" in file]

            if len(data_file_list) == 0 or len(label_file_list) == 0:
                raise Exception("No bottleneck features created, please use 'create_bottleneck_features()'.")

            # There should be an equal amount, and if there is assume that they will sort into the same order
            if len(data_file_list) != len(label_file_list):
                raise Exception("Feature data and label data do not match.")

            data_file_list.sort()
            label_file_list.sort()

            while len(data_file_list) > 0:
                features = np.load(open(join(data_directory, data_file_list.pop()), 'rb'))
                labels = np.load(open(join(data_directory, label_file_list.pop()), 'rb'))

                yield (features, labels)

    def __train_top_model(self, model_id, epochs, model_type, model):
        """Creates the top-model classifier using the provided model ID, creates the callbacks to save the model and
        tensorboard output, and trains the data using the bottleneck features already created.

        Arguments:
            model_id: The unique ID of the pre-trained model being used, to indicate the top model type.
            epochs: The number of epochs to train for.
            model_type: The type of model, used to select the correct bottleneck data for the generator.
            model: The top model to train
        """
        if exists(join(self.model_directory, model_id)):
            logging.warning("Cannot train: model already exists.")
        else:
            print("Beginning to train model: {} for {} epochs.".format(model_id, epochs))
            makedirs(join(self.model_directory, model_id))

            # Create the file with the model type
            file_name = "{}.model_type".format(model_type)
            try:
                open(join(self.model_directory, model_id, file_name), 'x')
            except FileExistsError:
                pass

            checkpoint_file = join(self.model_directory, model_id, "best-checkpoint.h5")
            model_history_file = join(self.model_directory, model_id, "history.pkl")
            tensorboard_folder = join(self.model_directory, model_id)

            save_best_model_cb = ModelCheckpoint(checkpoint_file,
                                                 monitor='val_acc',
                                                 verbose=0,
                                                 save_best_only=True,
                                                 mode='max',
                                                 period=1)

            tensorboard_cb = TensorBoard(log_dir=tensorboard_folder)

            number_training_batches = math.ceil(
                ((self.class_counts['number_instances'].sum() * (1-self.train_test_split)) / self.batch_size)
                * self.train_multiplier)

            number_validation_batches = math.ceil(
                ((self.class_counts['number_instances'].sum() * self.train_test_split) / self.batch_size))

            history = model.fit_generator(self.__bottleneck_data_generator(
                                          join(self.bottleneck_directory, model_type + "_" + "train")),
                                          steps_per_epoch=number_training_batches,
                                          epochs=epochs,
                                          class_weight=self.class_weights,
                                          callbacks=[save_best_model_cb, tensorboard_cb],
                                          validation_data=self.__bottleneck_data_generator(
                                          join(self.bottleneck_directory, model_type + "_" + "test")),
                                          validation_steps=number_validation_batches)

            # Save the model history object
            with open(model_history_file, 'wb') as f:
                pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)

    def train_provided_top_models(self, model_dict):
        """Trains the provided model for the provided number of epochs.

        Arguments:
            model_dict: A dictionary where keys are the model ID's, and the value a list of [model, model_type, epochs]
        """

        # Check the model type are supported:
        model_types = [model_list[1] for model_list in model_dict.values()]
        missing_models = [model_type for model_type in model_types if model_type not in self.available_models]

        if len(missing_models) > 0:
            logging.warning("The model dictionary is requesting an unavailable model(s): {}"
                            .format(missing_models))
        else:
            self.__create_directory_train_test_split()
            self.__create_bottleneck_features(model_types)
            for model_id in model_dict.keys():
                self.__train_top_model(model_id,
                                       model_dict.get(model_id)[2],
                                       model_dict.get(model_id)[1],
                                       model_dict.get(model_id)[0])

    # =================================================================================================================
    # Top Model Evaluation: Training Metrics of Accuracy and Loss
    # =================================================================================================================

    def evaluate_top_model_training(self):
        """Evaluate the training data for all the models that are available, by loading their history files and
        plotting the loss and accuracy measures."""

        self.model_list = self.__get_trained_models()

        model_training_data = {}
        for model_id in self.model_list:
            if exists(join(self.model_directory, model_id, "history.pkl")):
                with open(join(self.model_directory, model_id, "history.pkl"), "rb")as f:
                    model_history = pickle.load(f)
                    model_training_data[model_id] = model_history
        self.__compare_models(model_training_data)

    @staticmethod
    def __compare_models(model_histories):
        """Using the history dict objects of models, print a comparison of their performance.

        Arguments:
            model_histories: a dictionary object, where the history object created by Keras when fitting a model is the
                value.
        """
        line_colours = itertools.cycle(Colorblind8)

        compare_train_acc = figure(x_axis_label="Epoch", y_axis_label="Accuracy", plot_width=475, title="Training Accuracy")
        for current_model, current_colour in zip(model_histories.keys(), line_colours):
            compare_train_acc.line(
                range(len(model_histories.get(current_model)['acc'])),
                model_histories.get(current_model)['acc'],
                line_color=current_colour,
                legend=current_model)
        compare_train_acc.legend.location = "bottom_right"

        compare_test_acc = figure(x_axis_label="Epoch", y_axis_label="Accuracy", plot_width=475, title="Testing Accuracy")
        for current_model, current_colour in zip(model_histories.keys(), line_colours):
            compare_test_acc.line(
                range(len(model_histories.get(current_model)['val_acc'])),
                model_histories.get(current_model)['val_acc'],
                line_color=current_colour,
                legend=current_model)
            compare_test_acc.legend.location = "bottom_right"

        compare_train_loss = figure(x_axis_label="Epoch", y_axis_label="Loss", plot_width=475, title="Training Loss")
        for current_model, current_colour in zip(model_histories.keys(), line_colours):
            compare_train_loss.line(
                range(len(model_histories.get(current_model)['loss'])),
                model_histories.get(current_model)['loss'],
                line_color=current_colour,
                legend=current_model)
            compare_train_loss.legend.location = "top_right"

        compare_test_loss = figure(x_axis_label="Epoch", y_axis_label="Loss", plot_width=475, title="Testing Loss")
        for current_model, current_colour in zip(model_histories.keys(), line_colours):
            compare_test_loss.line(
                range(len(model_histories.get(current_model)['val_loss'])),
                model_histories.get(current_model)['val_loss'],
                line_color=current_colour,
                legend=current_model)
            compare_test_loss.legend.location = "top_right"

        show(row(compare_train_acc, compare_test_acc))
        show(row(compare_train_loss, compare_test_loss))

    # =================================================================================================================
    # Top Model Evaluation: Train/Test prediction analysis
    # =================================================================================================================

    def evaluate_model_prediction(self, method):
        """This calls upon the functions used to create predictions for the validation data set that wasn't
        used for training, to load them, and then to analyze them given the provided method.

        Arguments:
            method: The method to use to analyze the predictions, applied across all models.
        """
        self.model_list = self.__get_trained_models()
        self.__create_top_model_predictions()
        self.__get_top_model_predictions()

        if method == "accuracy":
            self.__create_accuracy_report()
        elif method == "confusion":
            self.__create_confusion_matrix()
        elif method == "heatmaps":
            self.__create_heatmaps()
        elif method == "extremes-examples":
            self.__create_extremes_image_plot()
        elif method == "bar-f1score":
            self.__show_fscore_bar()
        elif method == "line-f1score":
            self.__show_fscore_line()

    def __prediction_image_generator(self, model_id):
        """Generator used to iterate over the validation dataframe, loading a batch of 32 at a time, normalizing, and then
        returning to the calling function.  StopIteration is returned when there are no more observations to load.

        Arguments:
            model_id: the model the generator is creating image batches for.
        """

        local_images = self.predict_dataset

        for idx, row in local_images.iterrows():
            local_images.loc[idx, 'full_uri'] = join(row['directory'], row['image_id'])

        image_list = local_images['full_uri'].tolist()
        label_list = [self.class_labels[curr_label] for curr_label in local_images['class'].tolist()]

        while len(image_list) > 0:
            # Get the current subset of images and remove from the list being iterated over
            current_images = image_list[:32]
            current_labels = label_list[:32]
            image_list = image_list[32:]
            label_list = label_list[32:]

            return_images = []

            for curr_image_uri in current_images:
                return_image = load_img(curr_image_uri, target_size=(self.model_image_dimensions.get(model_id),
                                                                     self.model_image_dimensions.get(model_id)))
                return_image = img_to_array(return_image)
                return_images.append(return_image)
            return_images = np.array(return_images)

            if model_id in self.tf_models:
                preprocess_input(return_images, 'channels_last', 'tf')
            else:
                return_images /= 255.

            yield return_images, current_labels
        raise StopIteration()

    def __create_top_model_predictions(self):
        """Evaluate the performance on the training and test data for all the models that are available, by creating
        class predictions on the validation dataset not used for training."""

        for model_id in self.model_list:
            if not exists(join(self.model_directory, model_id, "predictions.csv")):
                if exists(join(self.model_directory, model_id, "best-checkpoint.h5")):
                    model_type = [type_file.split('.')[0]
                                  for type_file in listdir(join(self.model_directory, model_id))
                                  if ".model_type" in type_file]
                    if len(model_type) == 1:
                        model_type = model_type[0]
                        print("Creating predictions using {} for the validation data.".format(model_id))
                        pre_trained_model = self.get_model(model_type)
                        top_model = load_model(join(self.model_directory, model_id, "best-checkpoint.h5"))
                        model = Model(inputs=pre_trained_model.input,
                                      outputs=top_model(pre_trained_model.output))

                        predict_batch_generator = self.__prediction_image_generator(model_type)
                        model_predicted_classes = None

                        while True:
                            try:
                                x, y = next(predict_batch_generator)
                                if model_predicted_classes is None:
                                    model_predicted_classes = model.predict_on_batch(x)
                                else:
                                    model_predicted_classes = np.append(model_predicted_classes, model.predict_on_batch(x), axis=0)
                            except StopIteration:
                                np.savetxt(join(self.model_directory, model_id, "predictions.csv"), model_predicted_classes)
                                break
                    else:
                        logging.warning("Cannot create predictions for model {}, no model type information for base model.".format(model_id))
                else:
                    logging.warning("Cannot create predictions for model {}, no check-pointed best model from training.".format(model_id))
                    raise Exception("Model not trained.")
            else:
                logging.warning("Using previously created predictions for model: {}.".format(model_id))

    def __get_top_model_predictions(self):
        """Iterates over each set of predicted validation data for each model, and adds metadata about the predicted
        class used for analysis.  This function expects predictions to have already been created and written to file.
        It also expects the predictions in the file to be in the same order as the rows within the predict_dataset.
        """
        local_predictions = self.predict_dataset
        local_predicted_class_probabilities = pd.DataFrame()
        classes = self.image_df.drop_duplicates('class').loc[:, 'class'].sort_values(
            ascending=True).tolist()

        if self.predicted_dataset is None:
            for model_id in self.model_list:
                if exists(join(self.model_directory, model_id, "predictions.csv")):
                    model_predictions = np.loadtxt(join(self.model_directory, model_id, "predictions.csv"))

                    local_predictions[model_id + "-predicted-class"] = None
                    local_predictions[model_id + "-predicted-correct"] = None
                    local_predictions[model_id + "-predicted-distance"] = None
                    local_predictions[model_id + "-predicted-one-hot-label"] = None

                    for idx, row in local_predictions.iterrows():
                        local_predictions.loc[idx, model_id + "-predicted-one-hot-label"] = model_predictions[idx]

                        if row['one-hot-label'].argmax() == model_predictions[idx].argmax():
                            local_predictions.loc[idx, model_id + "-predicted-correct"] = True
                        else:
                            local_predictions.loc[idx, model_id + "-predicted-correct"] = False

                        distance = 1 - (model_predictions[idx][row['one-hot-label'].argmax()])
                        local_predictions.loc[idx, model_id + "-predicted-distance"] = distance

                        local_predictions.loc[idx, model_id + "-predicted-class"] = classes[model_predictions[idx].argmax()]

                        # Create another dataframe with the class probabilities to pivot with
                        top_classes_idx = np.argsort(model_predictions[idx])[-3:].tolist()
                        top_classes_idx.reverse()
                        for i in range(0, len(top_classes_idx)):
                            text = '{}: {}%'.format(classes[top_classes_idx[i]], str(round(model_predictions[idx][top_classes_idx[i]]*100, 2)))
                            temp_df = pd.DataFrame({'position': [i],
                                                    'model': [model_id],
                                                    'image_id': [row['image_id']],
                                                    'directory': [row['directory']],
                                                    'class': [row['class']],
                                                    'class_text': [text]})
                            local_predicted_class_probabilities = local_predicted_class_probabilities.append(
                                temp_df, ignore_index=True)

                else:
                    logging.warning("Cannot load predictions, as not all models have predictions created: {}.".format(model_id))
                    raise Exception("Prediction for model: {} doesn't exist.".format(model_id))

            self.predicted_dataset = local_predictions
            self.predicted_class_probabilities = local_predicted_class_probabilities
        else:
            logging.warning("Using previously loaded predictions.")

    def __create_accuracy_report(self):
        """Creates a simple text representation of the overall accuracy of each model on the validation dataset."""
        for model_id in self.model_list:
            y_test = self.predicted_dataset.loc[:, 'class']
            y_pred = self.predicted_dataset.loc[:, model_id + '-predicted-class']
            print("Classification Accuracy for {} is: {}".format(model_id, accuracy_score(y_test, y_pred)))

    def __create_heatmaps(self):
        """Creates the dataframe needed to generate a class heatmap for precision, recall, and f1-score by model, by
        class.
        """
        heatmap_data = pd.DataFrame()

        for model_id in self.model_list:
            y_test = self.predicted_dataset.loc[:, 'class']
            y_pred = self.predicted_dataset.loc[:, model_id + '-predicted-class']
            precision = precision_score(y_test, y_pred, average=None)
            recall = recall_score(y_test, y_pred, average=None)
            f1score = f1_score(y_test, y_pred, average=None)
            classes = self.predicted_dataset.drop_duplicates('class').loc[:, 'class'].sort_values(ascending=True).tolist()

            model_list = [model_id] * len(precision)
            temp_df = pd.DataFrame({'Model': model_list,
                                    'Class': classes,
                                    'precision': precision,
                                    'recall': recall,
                                    'f1score': f1score})
            heatmap_data = heatmap_data.append(temp_df, ignore_index=True)

        heatmap_data.loc[:, 'recall'] = heatmap_data.loc[:, 'recall'].apply(lambda x: x * 100)
        heatmap_data.loc[:, 'precision'] = heatmap_data.loc[:, 'precision'].apply(lambda x: x * 100)
        heatmap_data.loc[:, 'f1score'] = heatmap_data.loc[:, 'f1score'].apply(lambda x: x * 100)


        self.__create_single_heatmap(heatmap_data.loc[:, ('Model', 'Class', 'precision')],
                                     'Precision Performance of all Models: lower means too many false positives',
                                     'precision')

        self.__create_single_heatmap(heatmap_data.loc[:, ('Model', 'Class', 'recall')],
                                     'Recall Performance of all Models: lower means too many false negatives',
                                     'recall')

        self.__create_single_heatmap(heatmap_data.loc[:, ('Model', 'Class', 'f1score')],
                                     'F-Score Performance of all Models: used in place of Accuracy as imbalanced classes',
                                     'f1score')

    def __create_single_heatmap(self, df, title, val_col, image_height=300, rect_height=1):
        """Creates and displays heatmaps, given a value columns, for each class by model."""
        single_df = df.pivot(index='Model', columns='Class', values=val_col)
        #df = pd.DataFrame(recall_map.stack(), columns=['recall']).reset_index()

        source = ColumnDataSource(df)
        colors = PRGn11
        mapper = LinearColorMapper(palette=colors, low=0, high=100)

        p = figure(plot_width=900, plot_height=image_height, title=title,
                   x_range=list(single_df.columns), y_range=list(reversed(single_df.index)),
                   toolbar_location=None, tools="", x_axis_location="above")

        p.rect(x="Class", y="Model", width=1, height=rect_height, source=source,
               line_color='#e6f7ff', line_width=0.5, fill_color=transform(val_col, mapper))

        color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                             ticker=BasicTicker(desired_num_ticks=len(colors)),
                             formatter=PrintfTickFormatter(format="  %d%%"))

        p.add_layout(color_bar, 'right')

        p.axis.axis_line_color = None
        p.axis.major_tick_line_color = None
        p.axis.major_label_text_font_size = "5pt"
        p.axis.major_label_standoff = 0
        p.xaxis.major_label_orientation = 1.0

        show(p)

    def __create_confusion_matrix(self):
        """Creates a confusion matrix for each model for class predictions."""
        for model_id in self.model_list:
            y_test = self.predicted_dataset.loc[:, 'class']
            y_pred = self.predicted_dataset.loc[:, model_id + '-predicted-class']
            conf_matrix = confusion_matrix(y_test, y_pred)

            conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            conf_matrix *= 100
            conf_df = pd.DataFrame(conf_matrix)
            classes = self.predicted_dataset.drop_duplicates('class').loc[:, 'class'].sort_values(
                ascending=True).tolist()
            conf_df.index = classes
            conf_df.columns = classes
            conf_df.columns.name = 'Class'

            df = pd.DataFrame(conf_df.stack(), columns=['confnum']).reset_index()
            df.columns = ['Model', 'Class', 'normalised']
            print(df.loc[df['Model'] == df['Class']])
            #print(df)
            plot_title = "Confusion matrix for {}.".format(model_id)

            self.__create_single_heatmap(df, plot_title, 'normalised', 500)

    def __create_extremes_image_plot(self):
        """Show the eight best and worst performing images, with the actual class, and top three predictions."""

        print(
            """
            The best performing images are calculated as those that have the correct predicted class, ordered
            by those with the smallest distance between the prediction for the correct class and a perfect prediction.""")

        print(
            """
            The worst performing images are calculated as those that have the wrong predicted class, ordered
            by those with the largest distance between the prediction for the correct class and a perfect prediction.
            TODO: Maybe this should instead be how many classes are between the first prediction and the actual
            class prediction position?:""")

        for model_id in self.model_list:

            worst_performance = (self.predicted_dataset.loc[(~ self.predicted_dataset[model_id + '-predicted-correct'].astype(bool)), :]
                                 .sort_values(model_id + '-predicted-distance', ascending=False)
                                 .head(8)
                                 .loc[:, 'image_id']
                                 .tolist())[::-1]

            best_performance = (self.predicted_dataset.loc[(self.predicted_dataset[model_id + '-predicted-correct'].astype(bool)), :]
                                 .sort_values(model_id + '-predicted-distance', ascending=True)
                                 .head(8)
                                 .loc[:, 'image_id']
                                 .tolist())[::-1]

            p = plt.figure(0, figsize=(16, 16))
            img_grid = ImageGrid(p, 111, nrows_ncols=(2, 4), axes_pad=0.1)
            for i in range(0, 8):

                image_id = best_performance[i]

                prediction_data = self.predicted_class_probabilities.loc[
                    (self.predicted_class_probabilities['image_id'] == image_id) &
                    (self.predicted_class_probabilities['model'] == model_id),
                    ('class', 'directory', 'position', 'class_text')
                ].reset_index(drop=True)

                img = load_img(join(prediction_data.loc[0, 'directory'],
                                    best_performance[i]),
                               target_size=(300, 300))

                ax = img_grid[i]
                ax.imshow(img)

                prediction_one = prediction_data.loc[prediction_data['position'] == 0, 'class_text'].reset_index(drop=True).loc[0]
                prediction_two = prediction_data.loc[prediction_data['position'] == 1, 'class_text'].reset_index(drop=True).loc[0]
                prediction_three = prediction_data.loc[prediction_data['position'] == 2, 'class_text'].reset_index(drop=True).loc[0]

                all_predictions = prediction_one + "\n" +  prediction_two + "\n" + prediction_three

                image_title = "Class: {}\nPredicted:\n{}".format(prediction_data.loc[0, 'class'], all_predictions)

                title = AnchoredText(image_title,
                                     loc=2,
                                     frameon=False)
                ax.add_artist(title)
                title.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
                title.patch.set_alpha(0.5)

                ax.axis('off')
            print(
                """
                Best performing images for model: {}""".format(model_id))
            plt.show()

            p = plt.figure(0, figsize=(16, 16))
            img_grid = ImageGrid(p, 111, nrows_ncols=(2, 4), axes_pad=0.1)
            for i in range(0, 8):

                image_id = worst_performance[i]

                prediction_data = self.predicted_class_probabilities.loc[
                    (self.predicted_class_probabilities['image_id'] == image_id) &
                    (self.predicted_class_probabilities['model'] == model_id),
                    ('class', 'directory', 'position', 'class_text')
                ].reset_index(drop=True)

                img = load_img(join(prediction_data.loc[0, 'directory'],
                                    worst_performance[i]),
                               target_size=(300, 300))

                ax = img_grid[i]
                ax.imshow(img)

                prediction_one = prediction_data.loc[prediction_data['position'] == 0, 'class_text'].reset_index(drop=True).loc[0]
                prediction_two = prediction_data.loc[prediction_data['position'] == 1, 'class_text'].reset_index(drop=True).loc[0]
                prediction_three = prediction_data.loc[prediction_data['position'] == 2, 'class_text'].reset_index(drop=True).loc[0]

                all_predictions = prediction_one + "\n" +  prediction_two + "\n" + prediction_three

                image_title = "Class: {}\nPredicted:\n{}".format(prediction_data.loc[0, 'class'], all_predictions)

                title = AnchoredText(image_title,
                                     loc=2,
                                     frameon=False)
                ax.add_artist(title)
                title.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
                title.patch.set_alpha(0.5)

                ax.axis('off')
            print(
                """
                Worst performing images for model: {}""".format(model_id))
            plt.show()

    def __show_fscore_bar(self):
        """Creates a histogram of F1 scores per class per model."""
        classes = self.predicted_dataset.drop_duplicates('class').loc[:, 'class'].sort_values(ascending=True).tolist()

        for model_id in self.model_list:
            y_test = self.predicted_dataset.loc[:, 'class']
            y_pred = self.predicted_dataset.loc[:, model_id + '-predicted-class']
            f1score = (f1_score(y_test, y_pred, average=None) * 100).tolist()

            p = figure(y_range=(0, max(f1score)),
                       x_range=classes,
                       title="F1 Score by Class for: {}".format(model_id),
                       plot_width=950,
                       toolbar_location=None)

            p.vbar(x=classes,
                   top=f1score,
                   width=1)

            p.xaxis.axis_label = "Class"
            p.yaxis.axis_label = "F1 Score"
            p.axis.major_label_text_font_size = "8pt"
            p.axis.major_label_standoff = 5
            p.xaxis.major_label_orientation = 1.0
            show(p)

    def __show_fscore_line(self):
        """Creates a single plot of F1 scores as lines to allow easier comparison of performance between models."""

        # Select the largest number of colours, and if it is not large enough, raise an error
        if len(self.model_list) > 8:
            raise KeyError("Not enough colours in the ColorBlind palette to display the models.")
        else:
            colors = palettes.all_palettes['Colorblind'].get(8)
            colors = colors[:len(self.model_list)]
            color_idx = 0
            classes = self.predicted_dataset.drop_duplicates('class').loc[:, 'class'].sort_values(ascending=True).tolist()

            p = figure(y_range=(0, 100),
                       x_range=classes,
                       title="F1 Score by Class for All Models",
                       plot_width=950,
                       toolbar_location=None)

            for model_id in self.model_list:
                y_test = self.predicted_dataset.loc[:, 'class']
                y_pred = self.predicted_dataset.loc[:, model_id + '-predicted-class']
                f1score = (f1_score(y_test, y_pred, average=None) * 100).tolist()

                p.line(x=classes,
                       y=f1score,
                       color=colors[color_idx],
                       legend=model_id)
                p.circle(x=classes,
                         y=f1score,
                         color=colors[color_idx],
                         legend=model_id)

                color_idx += 1

            p.xaxis.axis_label = "Class"
            p.yaxis.axis_label = "F1 Score"
            p.axis.major_label_text_font_size = "8pt"
            p.axis.major_label_standoff = 5
            p.xaxis.major_label_orientation = 1.0
            show(p)

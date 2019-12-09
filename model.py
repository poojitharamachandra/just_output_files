# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by: Shangeth Rajaa, Zhengying Liu, Isabelle Guyon

"""An example of code submission for the AutoDL challenge.

It implements 3 compulsory methods ('__init__', 'train' and 'test') and
an attribute 'done_training' for indicating if the model will not proceed more
training due to convergence or limited time budget.

To create a valid submission, zip model.py together with other necessary files
such as Python modules/packages, pre-trained weights, etc. The final zip file
should not exceed 300MB.
"""

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import datetime
import logging
import numpy as np
import os
import sys
import time
import torch.utils.data as data_utils
import torch
import torch.nn as nn
import torchvision
import tensorflow as tf
import cv2
# example of global centering (subtract mean)
from numpy import asarray
from PIL import Image
from numpy import asarray
# seeding randomness for reproducibility
np.random.seed(42)
torch.manual_seed(1)
from sklearn.preprocessing import StandardScaler
from librosa.feature import melspectrogram
import librosa.display as d
from torchvision import models
from scipy import signal
from PIL import Image
from  torchvision import transforms
import sklearn.preprocessing
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import LabelBinarizer

# PyTorch Model class
class TorchModel(nn.Module):
    def __init__(self):
        ''' 3D CNN Model with no of CNN layers depending on the input size'''
        super(TorchModel, self).__init__()

        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
          param.requires_grad = False
        #30 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 9)
        #print(self.model)
        #self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.model(x)
        #out = self.log_softmax(out)
        return out


# PyTorch Dataset to get data from tensorflow Dataset.
class TFDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, session, num_samples):
        super(TFDataset, self).__init__()
        self.dataset = dataset
        self.session = session
        self.num_samples = num_samples
        self.next_element = None
        self.reset()

    def reset(self):
        dataset = self.dataset
        iterator = dataset.make_one_shot_iterator()
        self.next_element = iterator.get_next()
        return self

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        session = self.session if self.session is not None else tf.Session()
        try:
            example, label = session.run(self.next_element)
            example = np.squeeze(example, axis=0)
            #t = example.transpose(2,1,0)
            #print(t.shape)
            #print(example.shape)
            example = pre_process(example.transpose(1,0,2))
            #print(example.shape)
        except tf.errors.OutOfRangeError:
            self.reset()
            example, label = session.run(self.next_element)
            example = np.squeeze(example, axis=0)
            example = pre_process(example.transpose(1,0,2))
            #print(example.shape)
            #print("label while loading",np.argmax(label))
        return example, np.argmax(label)


def pre_process(x):
    #print(x.shape)
    mean = np.mean(x,axis=(0, 1, 2))
    std = np.std(x,axis=(0, 1, 2))
    x = cv2.resize(x,dsize=(224,224),interpolation=cv2.INTER_CUBIC)

    train_transforms = transforms.Compose([
            #transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #transforms.Normalize(mean=mean, std=std)
        ])

    x = train_transforms(x)
    x = (x-mean)/std
    return x


class Model():
    def __init__(self, metadata):
        """
        Args:
          metadata: an AutoDLMetadata object. Its definition can be found in
              AutoDL_ingestion_program/dataset.py
        """
        # Attribute necessary for ingestion program to stop evaluation process
        self.done_training = False
        self.metadata_ = metadata

        # Getting details of the data from meta data
        self.output_dim = self.metadata_.get_output_size()
        self.num_examples_train = self.metadata_.size()
        row_count, col_count = self.metadata_.get_matrix_size(0)
        channel = self.metadata_.get_num_channels(0)
        sequence_size = self.metadata_.get_sequence_size()

        self.num_train = self.metadata_.size()
        test_metadata_filename = self.metadata_.get_dataset_name() \
                                     .replace('train', 'test') + '/metadata.textproto'
        self.num_test = [int(line.split(':')[1]) for line
                         in open(test_metadata_filename, 'r').readlines()
                         if 'sample_count' in line][0]

        # Getting the device available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Device Found = ', self.device,
              '\nMoving Model and Data into the device...')

        # Attributes for preprocessing
        self.default_image_size = (224, 224)
        #self.default_sequence_length = 16000
        self.default_num_frames = 0
        self.default_shuffle_buffer = 100
        #self.default_num_frames = 16000

        if row_count == -1 or col_count == -1 :
            row_count = self.default_image_size[0]
            col_count = self.default_image_size[1]
        if sequence_size == -1: sequence_size = self.default_num_frames
        self.input_shape = (sequence_size, channel,row_count, col_count)
        print('\n\nINPUT SHAPE = ', self.input_shape)

        # getting an object for the PyTorch Model class for Model Class
        # use CUDA if available
        self.pytorchmodel = TorchModel()
        #model = TheModelClass(*args, **kwargs)
        print("current working directory :  "+os.getcwd())
        checkpoint = torch.load('../hpo/incumbent_small_dict.pth')
        #self.pytorchmodel = checkpoint['model']
        self.pytorchmodel.load_state_dict(checkpoint['state_dict'],strict=False)
        print('\nPyModel Defined\n')
        #print(self.pytorchmodel)
        self.pytorchmodel.to(self.device)
        self.pytorchmodel =  torch.nn.DataParallel(self.pytorchmodel)#, device_ids=[0, 1, 2, 3])

        # PyTorch Optimizer and Criterion
        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.pytorchmodel.parameters(),lr = 0.00012173481862884112)
        #self.optimizer.load_state_dict(checkpoint['optimizer'])
        #self.optimizer = torch.optim.SGD(self.pytorchmodel.parameters(), lr=1e-2)
        #self.scheduler = StepLR(self.optimizer, step_size=4, gamma=0.01)

        # Attributes for managing time budget
        # Cumulated number of training steps
        self.birthday = time.time()
        self.total_train_time = 0
        self.cumulated_num_steps = 0
        self.estimated_time_per_step = None
        self.total_test_time = 0
        self.cumulated_num_tests = 0
        self.estimated_time_test = None
        self.trained = False

        # PYTORCH
        # Critical number for early stopping
        self.num_epochs_we_want_to_train = 2000

        # no of examples at each step/batch
        self.train_batch_size = 64
        self.test_batch_size = 16

        # Tensorflow sessions to get the data from TFDataset
        self.train_session = tf.Session()
        self.test_session = tf.Session()

    def train(self, dataset, remaining_time_budget=None):
        """Train this algorithm on the tensorflow |dataset|.

        This method will be called REPEATEDLY during the whole training/predicting
        process. So your `train` method should be able to handle repeated calls and
        hopefully improve your model performance after each call.

        ****************************************************************************
        ****************************************************************************
        IMPORTANT: the loop of calling `train` and `test` will only run if
            self.done_training = False
          (the corresponding code can be found in ingestion.py, search
          'M.done_training')
          Otherwise, the loop will go on until the time budget is used up. Please
          pay attention to set self.done_training = True when you think the model is
          converged or when there is not enough time for next round of training.
        ****************************************************************************
        ****************************************************************************

        Args:
          dataset: a `tf.data.Dataset` object. Each of its examples is of the form
                (example, labels)
              where `example` is a dense 4-D Tensor of shape
                (sequence_size, row_count, col_count, num_channels)
              and `labels` is a 1-D Tensor of shape
                (output_dim,).
              Here `output_dim` represents number of classes of this
              multilabel classification task.

              IMPORTANT: some of the dimensions of `example` might be `None`,
              which means the shape on this dimension might be variable. In this
              case, some preprocessing technique should be applied in order to
              feed the training of a neural network. For example, if an image
              dataset has `example` of shape
                (1, None, None, 3)
              then the images in this datasets may have different sizes. On could
              apply resizing, cropping or padding in order to have a fixed size
              input tensor.

          remaining_time_budget: time remaining to execute train(). The method
              should keep track of its execution time to avoid exceeding its time
              budget. If remaining_time_budget is None, no time budget is imposed.
        """

        steps_to_train = self.get_steps_to_train(remaining_time_budget)
        if steps_to_train <= 0:
            logger.info("Not enough time remaining for training. " +
                        "Estimated time for training per step: {:.2f}, " \
                        .format(self.estimated_time_per_step) +
                        "but remaining time budget is: {:.2f}. " \
                        .format(remaining_time_budget) +
                        "Skipping...")
            self.done_training = True
        else:
            msg_est = ""
            if self.estimated_time_per_step:
                msg_est = "estimated time for this: " + \
                          "{:.2f} sec.".format(steps_to_train * self.estimated_time_per_step)
            logger.info("Begin training for another {} steps...{}".format(steps_to_train, msg_est))

            # If PyTorch dataloader for training set doesn't already exists, get the train dataloader
            if not hasattr(self, 'trainloader'):
                self.trainloader = self.get_dataloader(dataset, self.num_train, batch_size=self.train_batch_size)
            train_start = time.time()

            # Training loop
            self.trainloop(self.criterion, self.optimizer, steps=steps_to_train)
            train_end = time.time()

            # Update for time budget managing
            train_duration = train_end - train_start
            self.total_train_time += train_duration
            self.cumulated_num_steps += steps_to_train
            self.estimated_time_per_step = self.total_train_time / self.cumulated_num_steps
            logger.info("{} steps trained. {:.2f} sec used. ".format(steps_to_train, train_duration) + \
                        "Now total steps trained: {}. ".format(self.cumulated_num_steps) + \
                        "Total time used for training: {:.2f} sec. ".format(self.total_train_time) + \
                        "Current estimated time per step: {:.2e} sec.".format(self.estimated_time_per_step))

    def test(self, dataset, remaining_time_budget=None):
        """Test this algorithm on the tensorflow |dataset|.

        Args:
          Same as that of `train` method, except that the `labels` will be empty.
        Returns:
          predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
              here `sample_count` is the number of examples in this dataset as test
              set and `output_dim` is the number of labels to be predicted. The
              values should be binary or in the interval [0,1].
        """
        if self.done_training:
            return None

        if self.choose_to_stop_early():
            logger.info("Oops! Choose to stop early for next call!")
            self.done_training = True
        test_begin = time.time()
        if remaining_time_budget and self.estimated_time_test and \
                self.estimated_time_test > remaining_time_budget:
            logger.info("Not enough time for test. " + \
                        "Estimated time for test: {:.2e}, ".format(self.estimated_time_test) + \
                        "But remaining time budget is: {:.2f}. ".format(remaining_time_budget) + \
                        "Stop train/predict process by returning None.")
            return None

        msg_est = ""
        if self.estimated_time_test:
            msg_est = "estimated time: {:.2e} sec.".format(self.estimated_time_test)
        logger.info("Begin testing..." + msg_est)

        # If PyTorch dataloader for training set doen't already exists, get the test dataloader
        if not hasattr(self, 'testloader'):
            self.testloader = self.get_dataloader_test(dataset, self.num_test,
                                                       self.test_batch_size)

        # get predictions from the test loop
        predictions = self.testloop(self.testloader)

        test_end = time.time()
        # Update some variables for time management
        test_duration = test_end - test_begin
        self.total_test_time += test_duration
        self.cumulated_num_tests += 1
        self.estimated_time_test = self.total_test_time / self.cumulated_num_tests
        logger.info("[+] Successfully made one prediction. {:.2f} sec used. ".format(test_duration) + \
                    "Total time used for testing: {:.2f} sec. ".format(self.total_test_time) + \
                    "Current estimated time for test: {:.2e} sec.".format(self.estimated_time_test))
        return predictions

    ##############################################################################
    #### Above 3 methods (__init__, train, test) should always be implemented ####
    ##############################################################################

    def preprocess_tensor_4d(self, tensor_4d):
        """Preprocess a 4-D tensor (only when some dimensions are `None`, i.e.
        non-fixed). The output tensor wil have fixed, known shape.
        Args:
          tensor_4d: A Tensor of shape
              [sequence_size, row_count, col_count, num_channels]
              where some dimensions might be `None`.
        Returns:
          A 4-D Tensor with fixed, known shape.
        """
        tensor_4d_shape = tensor_4d.shape
        logger.info("Tensor shape before preprocessing: {}".format(tensor_4d_shape))

        if tensor_4d_shape[0] > 0 and tensor_4d_shape[0] < 10:
            num_frames = tensor_4d_shape[0]
        else:
            num_frames = self.default_num_frames
        if tensor_4d_shape[1] > 0:
            new_row_count = tensor_4d_shape[1]
        else:
            new_row_count = self.default_image_size[0]
        if tensor_4d_shape[2] > 0:
            new_col_count = tensor_4d_shape[2]
        else:
            new_col_count = self.default_image_size[1]

        if not tensor_4d_shape[0] > 0:
            logger.info("Detected that examples have variable sequence_size, will " +
                        "randomly crop a sequence with num_frames = " +
                        "{}".format(num_frames))
            tensor_4d = crop_time_axis(tensor_4d, num_frames=num_frames)
        if not tensor_4d_shape[1] > 0 or not tensor_4d_shape[2] > 0:
            logger.info("Detected that examples have variable space size, will " +
                        "resize space axes to (new_row_count, new_col_count) = " +
                        "{}".format((new_row_count, new_col_count)))
            tensor_4d = resize_space_axes(tensor_4d,
                                          new_row_count=new_row_count,
                                          new_col_count=new_col_count)
        logger.info("Tensor shape after preprocessing: {}".format(tensor_4d.shape))
        return tensor_4d

    def get_mean_and_std(self, dataloader):
        pop_mean = []
        pop_std = []
        print('Calculating mean and std of training dataset')
        for data in dataloader:
            np_image = data[0].numpy()

            batch_mean = np.mean(np_image, axis=(0, 1 ,2))
            batch_std = np.std(np_image, axis=(0, 1, 2), ddof=1)

            pop_mean.append(batch_mean)
            pop_std.append(batch_std)

        pop_mean = np.array(pop_mean).mean(axis=0)
        pop_std = np.array(pop_std).mean(axis=0)

        return [pop_mean, pop_std]


    def get_dataloader(self, tf_dataset, num_images, batch_size):
        ''' Get the training PyTorch dataloader
        Args:
          tf_dataset: Tensorflow Dataset which is given in train function
          num_images : number of examples in train data
          batch_size : batch_size for training set

        Return:
          dataloader: PyTorch Training Dataloader
        '''
        tf_dataset = tf_dataset.map(lambda *x: (self.preprocess_tensor_4d(x[0]), x[1]))
        train_dataset = TFDataset(tf_dataset, self.train_session, num_images)
        #print(num_images)
        #print(train_dataset.shape)
        #normalize = nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #train_dataset= normalize(torch.unsqueeze(train_dataset,0))
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            drop_last=False
        )
        # batch normalization
        '''mean, std = self.get_mean_and_std(dataloader)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)'''

        #transform using mean and std
        

        return dataloader

    def get_dataloader_test(self, tf_dataset, num_images, batch_size):
        ''' Get the test PyTorch dataloader
        Args:
          tf_dataset: Tensorflow Dataset which is given in test function
          num_images : number of examples in test data
          batch_size : batch_size for test set

        Return:
          dataloader: PyTorch Test Dataloader
        '''
        tf_dataset = tf_dataset.map(lambda *x: (self.preprocess_tensor_4d(x[0]), x[1]))
        dataset = TFDataset(tf_dataset, self.test_session, num_images)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        #normalize = nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #dataset= normalize(dataset)
        #mean, std = self.get_mean_and_std(dataloader)
        #dataset = TFDataset_dummy(tf_dataset, self.test_session, num_images,mean,std)
        #dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        return dataloader

    def trainloop(self, criterion, optimizer, steps):
        ''' Training loop with no of given steps
        Args:
          criterion: PyTorch Loss function
          Optimizer: PyTorch optimizer for training
          steps: No of steps to train the model

        Return:
          None, updates the model parameters
        '''
        #self.pytorchmodel.train()
        self.pytorchmodel.eval()
        data_iterator = iter(self.trainloader)
        for i in range(steps):
            try:
                images, labels = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.trainloader)
                images, labels = next(data_iterator)

            with torch.cuda.device(0):
                #device = torch.device('cuda:0')
                #images = images.float().to(device)
                #labels = labels.float().to(device,dtype=torch.int64)
                images = images.float().cuda()
                labels = labels.float().cuda()
                labels = labels.to(dtype=torch.int64)
            #if hasattr(self, 'scheduler'):
                #self.scheduler.step()
   
            optimizer.zero_grad()

            log_ps = self.pytorchmodel(images)
            loss = criterion(log_ps, labels)
            if hasattr(self, 'scheduler'):
                self.scheduler.step(loss)
            loss.backward()
            optimizer.step()

    def get_steps_to_train(self, remaining_time_budget):
        """Get number of steps for training according to `remaining_time_budget`.

        The strategy is:
          1. If no training is done before, train for 10 steps (ten batches);
          2. Otherwise, estimate training time per step and time needed for test,
             then compare to remaining time budget to compute a potential maximum
             number of steps (max_steps) that can be trained within time budget;
          3. Choose a number (steps_to_train) between 0 and max_steps and train for
             this many steps. Double it each time.
        """
        if not remaining_time_budget:  # This is never true in the competition anyway
            remaining_time_budget = 1200  # if no time limit is given, set to 20min

        if not self.estimated_time_per_step:
            steps_to_train = 10
        else:
            if self.estimated_time_test:
                tentative_estimated_time_test = self.estimated_time_test
            else:
                tentative_estimated_time_test = 50  # conservative estimation for test
            max_steps = int((remaining_time_budget - tentative_estimated_time_test) / self.estimated_time_per_step)
            max_steps = max(max_steps, 1)
            if self.cumulated_num_tests < np.log(max_steps) / np.log(2):
                steps_to_train = int(2 ** self.cumulated_num_tests)  # Double steps_to_train after each test
            else:
                steps_to_train = 0
        return steps_to_train

    def testloop(self, dataloader):
        '''
        Args:
          dataloader: PyTorch test dataloader

        Return:
          preds: Predictions of the model as Numpy Array.
        '''
        labels = torch.tensor((),dtype=torch.long)
        preds = []
        
        with torch.no_grad():
            self.pytorchmodel.eval()
            for images, label in dataloader:
                if torch.cuda.is_available():
                    with torch.cuda.device(0):
                        images = images.float().cuda()
                else:
                    images = images.float()
                log_ps = self.pytorchmodel(images)
                #print("label",label)
                #print("type of labels",type(label))
                #pred = self.pytorchmodel(images)
                #pred = pred > 0.5
                #print("log ps",log_ps)
                #max_p = log_ps.max(dim=1)[0]
                
                max_p,_ = torch.max(log_ps.data,1)
                max_p = torch.reshape(max_p , (log_ps.shape[0],1))
                labels=torch.cat((labels,label.cpu()),0)
                #print("max value",max_p)
                #print(log_ps.shape)
                #one_hot = torch.FloatTensor(log_ps.shape)
                #one_hot.zero_()
                #one_hot.scatter_(0, max_p, 1)
                #p =  np.argmax(np.squeeze(log_ps.cpu()))
                #max_p =  torch.max(np.squeeze(log_ps.cpu()))
                #print("class :",p.item())
                #pred = torch.sigmoid(log_ps).data > 0.5
                #pred = np.exp(log_ps.cpu()) > 0.5
                pred = log_ps == max_p
                preds.append(pred.cpu().numpy())
                #pred = torch.exp(log_ps).data > 1
                #print(pred)
                #pred = np.argmax(np.squeeze(log_ps.cpu()))
                #pred = log_ps == max_p.item()
                #pred = log_ps.max(dim=1)[1]
                #print("prediction calc.",pred)
                #preds.append(pred.cpu().numpy())
        preds = np.vstack(preds)
        #print("type of predictions :",type(preds))

        return preds

    def choose_to_stop_early(self):
        """The criterion to stop further training (thus finish train/predict
        process).
        """
        # return self.cumulated_num_tests > 10 # Limit to make 10 predictions
        # return np.random.rand() < self.early_stop_proba
        batch_size = self.train_batch_size
        num_examples = self.metadata_.size()
        num_epochs = self.cumulated_num_steps * batch_size / num_examples
        logger.info("Model already trained for {} epochs.".format(num_epochs))
        return num_epochs > self.num_epochs_we_want_to_train  # Train for at least certain number of epochs then stop


#### Other helper functions
def crop_time_axis(tensor_4d, num_frames, begin_index=None):
    """Given a 4-D tensor, take a slice of length `num_frames` on its time axis.
    Args:
      tensor_4d: A Tensor of shape
          [sequence_size, row_count, col_count, num_channels]
      num_frames: An integer representing the resulted chunk (sequence) length
      begin_index: The index of the beginning of the chunk. If `None`, chosen
        randomly.
    Returns:
      A Tensor of sequence length `num_frames`, which is a chunk of `tensor_4d`.
    """
    # pad sequence if not long enough
    pad_size = tf.maximum(num_frames - tf.shape(tensor_4d)[0], 0)
    padded_tensor = tf.pad(tensor_4d, ((0, pad_size), (0, 0), (0, 0), (0, 0)))
    # If not given, randomly choose the beginning index of frames
    if not begin_index:
        maxval = tf.shape(padded_tensor)[0] - num_frames + 1
        begin_index = tf.random.uniform([1],
                                        minval=0,
                                        maxval=maxval,
                                        dtype=tf.int32)
        begin_index = tf.stack([begin_index[0], 0, 0, 0], name='begin_index')
    sliced_tensor = tf.slice(padded_tensor,
                             begin=begin_index,
                             size=[num_frames, -1, -1, -1])
    return sliced_tensor


def resize_space_axes(tensor_4d, new_row_count, new_col_count):
    """Given a 4-D tensor, resize space axes to have target size.
    Args:
      tensor_4d: A Tensor of shape
          [sequence_size, row_count, col_count, num_channels].
      new_row_count: An integer indicating the target row count.
      new_col_count: An integer indicating the target column count.
    Returns:
      A Tensor of shape [sequence_size, target_row_count, target_col_count].
    """
    resized_images = tf.image.resize_images(tensor_4d,
                                            size=(new_row_count, new_col_count))
    return resized_images


def get_logger(verbosity_level):
    """Set logging format to something like:
         2019-04-25 12:52:51,924 INFO model.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


logger = get_logger('INFO')

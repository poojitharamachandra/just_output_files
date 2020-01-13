import torch

import torch.nn as nn

import torchvision.transforms as transforms

import torchvision.datasets as dsets

from torch.utils.data import TensorDataset

from torch.autograd import Variable

from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import models

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
#from smac.facade.smac_facade import SMAC
from smac.facade.smac_hpo_facade import SMAC4HPO as SMAC
from libscores import read_array, sp, ls, mvmean, tiedrank, _HERE, get_logger



class TorchModel(nn.Module):
    def __init__(self):
        ''' 3D CNN Model with no of CNN layers depending on the input size'''
        super(TorchModel, self).__init__()

        self.model = models.resnet50(pretrained=True)
        #for param in self.model.parameters():
        #  param.requires_grad = False
        # 30 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 30)
        #print(self.model)
        #self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.model(x)
        #out = self.log_softmax(out)
        return out


import os

import requests
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelBinarizer



class Spectrogram():
  
  def __init__():
    pass

  def get_mean_and_std(train_path, num_workers,batch_size):
    """
    Calculate mean and std dev. of training dataset.

    Returns:
    pop_mean - list containing mean at different channels
    pop-std - list containing std. dev at different channels

    """
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])
    dataset = dsets.ImageFolder(root=train_path, transform=tfms)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    pop_mean = []
    pop_std = []

    print('Calculating mean and std of training dataset')
    for data in dataloader:
        np_image = data[0].numpy()

        batch_mean = np.mean(np_image, axis=(0, 2, 3))
        batch_std = np.std(np_image, axis=(0, 2, 3), ddof=1)

        pop_mean.append(batch_mean)
        pop_std.append(batch_std)

    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std = np.array(pop_std).mean(axis=0)

    return [pop_mean, pop_std]
  
  def get_data_loaders(train_path,train_transforms=None,num_workers=4):
    """
    Return train, validation and test dataloaders.

    Parameters:
        train_path - path to training data folder
        valid_path - path to validation data folder
        test_path - path to test data folder
        batch_size - batch size to use for dataloaders
        num_workers - num_workers to use for dataloaders
        train_transforms - transformations to apply on training data
        valid_transforms - transformations to apply on validation data
        test_transforms - transformations to apply on test data

    """
    mean, std = Spectrogram.get_mean_and_std(train_path, num_workers,batch_size=64)
    # get_mean_and_std() returns numpy array of mean and std
    # passing numpy array to normalize changes datatype of torch tensor
    # to numpy becuase of subtraction and division
    # This causes error when getting data from dataloader
    # Hence, converting numpy array to torch tensors
    #mean = torch.Tensor(mean)
    #std = torch.Tensor(std)
    #print(f'Mean:{mean}')
    #print(f'Std:{std}')
    '''train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            #transforms.Normalize(mean=mean, std=std)
        ])
    train_dataset = dsets.ImageFolder(
        root=train_path, transform=train_transforms)
    
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)

    images=torch.tensor((),dtype=torch.float32)
    #images = np.empty(0,dtype=np.float64)


    for data in train_data_loader:
        #print(data[0].size())
        images = torch.cat((images,data[0]),0)

        #print(data)

    scaler = StandardScaler()

    scaler.partial_fit(images.numpy())

    mean = scaler.mean_
    std = scaler.scale_'''



    if train_transforms:
        train_transforms = train_transforms
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transforms.Normalize(mean=mean, std=std)
        ])

    train_dataset = dsets.ImageFolder(
        root=train_path, transform=train_transforms)

    print("class mapping")
    print(train_dataset.class_to_idx)


    return train_dataset

batch_size = 64

print("loading data set from ./dataset/train")
train_dataset = Spectrogram.get_data_loaders(train_path='./dataset/train')
test_dataset = Spectrogram.get_data_loaders(train_path='./dataset/validate/generated/spectrograms')
test_holdout_dataset = Spectrogram.get_data_loaders(train_path='./dataset/validate/generated/spectrograms')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,num_workers=4,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size,num_workers=4,
                                          shuffle=False )

val_loader = torch.utils.data.DataLoader(dataset=test_holdout_dataset,  batch_size=batch_size,num_workers=4,
                                          shuffle=False )


criterion = nn.CrossEntropyLoss()

def prepare_cnn(cfg):
  print(cfg)
  cfg = {k : cfg[k] for k in cfg if cfg[k]}
  print(cfg)
  run_cnn(**cfg)


f= open("./results.txt","w+")
d= open("./data_to_plot.txt","w+")


def train(train_loader,test_loader,model,optimizer):
          model.train()
          for i, (images, labels) in enumerate(train_loader):
              if torch.cuda.is_available():
                  images = Variable(images.cuda())
                  labels = Variable(labels.cuda())

              else:
                  images = Variable(images)
                  labels = Variable(labels)

              optimizer.zero_grad()
              outputs = model(images)
              loss = criterion(outputs, labels)
              loss.backward()
              optimizer.step()

          roc=test(test_loader,model)

          return loss.item(),roc

def test(test_loader,model):
        model.eval()
        label=torch.tensor((),dtype=torch.int)
        pred = []
        for images, labels in test_loader:
            if torch.cuda.is_available():
                 images = Variable(images.cuda())
            else:
                 images = Variable(images)
            outputs = model(images)
            max_value, predicted_class = torch.max(outputs.data, 1)
            max_value = torch.reshape(max_value , (outputs.shape[0],1))


            onehotlabels = np.zeros((labels.shape[0], 30),dtype=int)
            onehotlabels[np.arange(labels.shape[0]), labels] = 1
            onehotlabels = torch.from_numpy(onehotlabels).int()
            
            label=torch.cat((label,onehotlabels),0)
            
            predicted = outputs == max_value
            pred.append(predicted.cpu().numpy())
            
        pred = np.vstack(pred)
        y_test = label.numpy()
        
        label_num = y_test.shape[1]
        auc = np.empty(label_num)
        for k in range(label_num):
            r_ = tiedrank(pred[:, k])
            s_ = y_test[:, k]
            if sum(s_) == 0: print("WARNING: no positive class example in class {}"\
                                                                 .format(k + 1))
            npos = sum(s_ == 1)
            nneg = sum(s_ < 1)
            auc[k] = (sum(r_[s_ == 1]) - npos * (npos + 1) / 2) / (nneg * npos)
        
        return 2 * mvmean(auc) - 1

        


def run_cnn(cfg):
    
  print("Running:",cfg)
  
  learning_rate = cfg["lr"]
  
  epochs=cfg["epoch"]
 
  model = TorchModel()

  if torch.cuda.is_available():
      model.cuda()
      
  #batch_size=cfg["batch_size"]
  optim = cfg["optimizer"]
  if optim == "adam":
     wt_dc = 0.0001
     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                                                       betas=(0.9, 0.999), eps=1e-08, weight_decay=wt_dc, amsgrad=False)
  elif optim == "sgd":
     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  else:
     print("optimizer not found")
  
  for epoch in range(epochs):      
          loss,area_under_curve =  train(train_loader,test_loader,model,optimizer)
          print('Epoch: {}. Loss: {}. area_under_curve: {}'.format(epoch, loss, area_under_curve))
          f.write('Epoch: {}. Loss: {}. area_under_curve: {}'.format(epoch, loss, area_under_curve))
  print("[",learning_rate,batch_size,area_under_curve,"]")
  print("Loss:",loss)
  output=[lr,batch_size,area_under_curve]
  d.write("[")
  d.write("%f"%learning_rate)
  d.write("%d"%batch_size)
  d.write("%f"%area_under_curve)
  d.write("]")
  d.write("\n")

  return loss


def run_cnn_again(cfg,is_incumbent=False):

  print("Running:",cfg)

  learning_rate = cfg["lr"]

  epochs=cfg["epoch"]

  model = TorchModel()


  if torch.cuda.is_available():
      model.cuda()

  #batch_size=cfg["batch_size"]

  optim = cfg["optimizer"]
  if optim == "adam":
        #wt_dc=cfg["weight_decay"]
        wt_dc = 0.0001
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                             betas=(0.9, 0.999), eps=1e-08, weight_decay=wt_dc, amsgrad=False)
  elif optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  else:
      print("optimizer not found")


  for epoch in range(epochs):
          loss,accuracy =  train(train_loader,val_loader,model,optimizer)
          print('Epoch: {}. Loss: {}. area_under_curve: {}'.format(epoch, loss, accuracy))
          f.write('Epoch: {}. Loss: {}. area_under_curve: {}'.format(epoch, loss, accuracy))
  print("[",learning_rate,batch_size,accuracy,epochs,"]")
  print("Loss:",loss)
  output=[lr,batch_size,accuracy,epochs]
  d.write("[")
  d.write("%f"%learning_rate)
  d.write("%d"%batch_size)
  d.write("%f"%accuracy)
  d.write("%f"%epochs)
  d.write("]")
  d.write("\n")
  print("saving model to ",os.getcwd())
  torch.save({'state_dict':model.state_dict(),'optimizer' : optimizer.state_dict()}, os.getcwd()+"/incumbent_mean_epoch25_without_decay_dict.pth")
  torch.save(model,os.getcwd()+"/incumbent_mean_epoch25_without_decay_model.pth")


  return loss

cs = ConfigurationSpace()

optim = CategoricalHyperparameter("optimizer",["adam","sgd"],default_value="adam")
cs.add_hyperparameter(optim)

lr = UniformFloatHyperparameter("lr", 0.0001, 0.001, default_value=0.001)
cs.add_hyperparameter(lr)

#wt = UniformFloatHyperparameter("weight_decay", 0.0001, 0.1, default_value=0.001)
#cs.add_hyperparameter(wt)


#cs.add_condition(InCondition(child=wt, parent=optim, values=["adam"]))


epoch = UniformIntegerHyperparameter("epoch",1,50,default_value=11)
cs.add_hyperparameter(epoch)

#batch_size = CategoricalHyperparameter("batch_size", [32,64,16,8], default_value=32)
#cs.add_hyperparameter(batch_size)

# Scenario object
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 25,  # maximum function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "true",
                     #"abort_on_first_run_crash": "false"
                     })


# Optimize, using a SMAC-object
#print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
        tae_runner=run_cnn)

print("Searching for incumbent config!")

smac.solver.intensifier.tae_runner.use_pynisher = False

incumbent = smac.optimize()

print("incumbent configuration:")
f.write("incumbent configuration:\n")


inc_value = run_cnn_again(incumbent,is_incumbent=True)

print("Optimized loss Value: %.2f" % (inc_value))

f.close()
d.close()

f= open("./results.txt","w+")
d= open("./data_to_plot.txt","w+")
inc_value = run_cnn_again(incumbent,is_incumbent=True)

print("Optimized loss Value: %.2f" % (inc_value))



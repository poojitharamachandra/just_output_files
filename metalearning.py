import os
from scipy import stats
from dataset import AutoSpeechDataset
import librosa
from tensorflow.python.keras.preprocessing import sequence
import tensorflow as tf
import numpy as np
import cupy as cp
import torch
import minpy
from mxnet import nd
from numpy import percentile
from sklearn.cluster import KMeans
print(torch.version.cuda)
def calc_covariance(x):
    x = cp.asarray(x)
    N = x.shape[2]
    m1 = x - x.sum(2,keepdims=1)/N
    out = np.einsum('ijk,ilk->ijl',m1,m1)  / (N - 1)
    return out

def pad_seq(data,pad_len):
    return sequence.pad_sequences(data,maxlen=pad_len,dtype='float32',padding='pre')


def extract_mfcc(data,sr=16000):
    results = []
    length =[]
    for d in data:
        r = librosa.feature.mfcc(d,sr=16000,n_mfcc=13)
        r = r.transpose()
        length.append(r.shape[0])
        #if r.shape[0] < 2000:
        results.append(r)
    l = remove_outliers(length)
    #print(l)
    results = [x for x in results if x.shape[0] in l]
    return results

'''def remove_outliers(data):
    print(data)
    print(len(data))
    X = np.asarray(data).reshape(-1,1)
    k = KMeans(n_clusters=2, random_state=0).fit(X)
    idx = np.squeeze(np.argwhere(k.labels_))
    print(idx)
    print(len(idx))
    data_o = data[idx]
    return data_o'''

'''def remove_outliers(data):
    #remove outliers with extremely lnong sequence which causes problems in padding
    q3,q1=percentile(data , 75),percentile(data , 25)
    iqr = q3-q1
    cut_off = 1.5*iqr
    lower = q1 - cut_off
    upper = q3 + cut_off
    print("upper : ", upper)
    print("lower  : ", lower)
    data_o = [x for x in data if x<=upper and x>=lower]
    outliers = [x for x in data if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    print(len(data_o))
    print(data_o)
    return data_o'''
def remove_outliers(data):
    # calculate summary statistics
    data_mean, data_std = np.mean(data), np.std(data)
    # identify outliers
    cut_off = data_std * 2
    lower, upper = data_mean - cut_off, data_mean + cut_off
    # identify outliers
    outliers = [x for x in data if x < lower or x > upper]
    p = len(outliers)/len(data)
    print('Identified outliers: %d' % len(outliers))
    print('% of outliers : ',p)
    print(outliers)
    d = max(data)*len(data)
    #print(data)
    # remove outliers
    if p < 0.05 and d > 32000000: 
        outliers_removed = [x for x in data if x >= lower and x <= upper]
        #print(outliers_removed)
        print("outliers removed!")
        return  outliers_removed
    else:
        return data

dirs = os.listdir("../sample_data/challenge/test")
#dirs.remove('starcraft')
#dirs.remove('music_env_speech')
#dirs.remove('flickr')
#dirs.remove('test')
print(dirs)
data_to_write=[]
#dirs = ['data04','data05']
#dirs=['flickr']
#dirs = ['espeak-numbers']
#dirs=['music_env_speech']
for d in dirs:
    p = os.path.join("../sample_data/challenge/test",d)
    D = AutoSpeechDataset(os.path.join(p,d+".data"))
    metadata = D.get_metadata()
    D.read_dataset()
    output_dim = D.get_class_num()
    D_train = D.get_train()
    D_test = D.get_test()
    num_test_samples = metadata['test_num']
    num_train_samples = metadata['train_num']
    sampling_rate = 16000
    for k,v in metadata.items():
        if k == 'sampling_rate':
            sampling_rate = metadata['sampling_rate']


    train_data, train_labels = D_train
    test_data = D_test
    train_features = extract_mfcc(train_data)
    test_features = extract_mfcc(D_test)

    print(d)

    print(type(test_features))
    print(type(train_features))

    print("extracted mfcc")

    train_max_len = max([len(_) for _ in train_features])
    test_max_len = max([len(_) for _ in test_features])
    max_len = max(train_max_len, test_max_len)  # for CNN variants we need max sequence length in advance
    print("max len : ",max_len)

    import matplotlib.pyplot as plt
    train_len = [len(_) for _ in train_features]
    #test_len = [len(_) for _ in test_features]
    plt.hist(train_len,bins=50)
    plt.savefig("train_"+d+".png")
    #plt.xlim(0,2000)


    train_features = pad_seq(train_features, max_len)  # padding at the beginning of the input
    test_features = pad_seq(test_features, max_len)

    #test_features = np.asarray(test_features)
    #train_features = np.asarray(train_features)

    #test_features = tf.convert_to_tensor(test_features)
    #train_features = tf.convert_to_tensor(train_features)
    #test_features = torch.stack(test_features)
    #train_features = torch.stack(train_features)


    print(type(test_features))
    print(type(train_features))

    print(test_features.shape)
    print(train_features.shape)


    #train_features = np.asarray(train_features, dtype=np.float32)
    #test_features = np.asarray(test_features, dtype=np.float32)

    #print(type(test_features))
    #print(type(train_features))

    #print("sequence padding done")

    x_data = np.concatenate((train_features,test_features),axis=0)

    #n , m , mean , variance,skewness, kurtosis = stats.describe(x_data,axis=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_data = torch.from_numpy(x_data).float().to(device)

    mean = torch.mean(x_data,0,False)
    diffs = x_data - mean
    var = torch.mean(torch.pow(diffs, 2.0),0,False)
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0),0,False)
    kurtosis = torch.mean(torch.pow(zscores, 4.0),0,False) - 3.0
    
    mean_var = torch.mean(var,0,False)
    mean_var[torch.isnan(mean_var)]=0
    mean_kurtosis = torch.mean(kurtosis,0,False)
    mean_kurtosis[torch.isnan(mean_kurtosis)]=0
    mean_skew = torch.mean(skews,0,False)
    mean_skew[torch.isnan(mean_skew)]=0
    mean_mean=torch.mean(mean,0,False)
    mean_mean[torch.isnan(mean_mean)]=0

    data={}

    data['name']=d
    #print(mean_mean.data)
    #print(mean_mean.shape)
    
    for i in range(1,14):
        st = "mean_"+str(i)
        data[st]=mean_mean[i-1].item()

    for i in range(1,14):
        st = "var_"+str(i)
        data[st]=mean_var[i-1].item()

    for i in range(1,14):
        st = "skew_"+str(i)
        data[st]=mean_skew[i-1].item()

    for i in range(1,14):
        st = "kurtosis_"+str(i)
        data[st]=mean_kurtosis[i-1].item()

    #data['mean_mfcc']=mean_mean.data
    #data['var_mfcc']=mean_var.data
    #data['skew_mfcc']=mean_skew.data
    #data['kurtosis_mfcc']=mean_kurtosis.data

    

    print('stats for mfcc dimension :')
    print("mean: ",mean_mean)
    print("shape of mean : ",mean.shape)
    print("variance: ",mean_var)
    print("skewness: ",mean_skew)
    print("kurtosis: ",mean_kurtosis)


    #print("covariance: ",calc_covariance(x_data))
    #print(type(x_data))
    #print(type(x_data))
    #x_data = nd.array(x_data)
    print("after using cuda: ")
    print(type(x_data))

    #entr = stats.entropy(x_data)
    mean = torch.mean(x_data)
    diffs = x_data - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtosis = torch.mean(torch.pow(zscores, 4.0)) - 3.0

    data['mean']=mean.item()
    data['var']=var.item()
    data['skew']=skews.item()
    data['kurtosis']=kurtosis.item()


    '''n , m , mean , variance,skewness, kurtosis = stats.describe(x_data,axis=None)'''

    print('stats accross data :')
    print("mean: ",mean.item())
    #print('shape of mean: ',mean.shape)
    print("variance: ",var.item())
    #print('shape of variance: ',variance.shape)
    print("skewness: ",skews.item())
    #print('shape of skewness: ',skewness.shape)
    print("kurtosis: ",kurtosis.item())
    #print('shape of kurtosis: ',kurtosis.shape)
    print(sampling_rate)
    print(num_train_samples)
    print(num_test_samples)
    print(output_dim)

    data['sampling_rate']=sampling_rate
    data['num_train_samples']=num_train_samples
    data['num_test_samples']=num_test_samples
    data['output_dim']=output_dim

    data_to_write.append(data)


#print(data_to_write)
#import xlsxwriter

# Create a workbook and add a worksheet.
#workbook = xlsxwriter.Workbook('data.xlsx')
#worksheet = workbook.add_worksheet()
row = 0
col = 0

heading = ['name','mean','var','skew','kurtosis','sampling_rate','num_train_samples','num_test_samples','output_dim']
l = ['mean','var','skew','kurtosis']
for word in l:
    c =0
    while c < 13:
        c+=1
        w=word+"_"+str(c)
        heading.append(w)
print(heading)

'''while col < 61:
      worksheet.write(0 , col , heading[col])
      col+=1'''

row = 0
col = 0
dd=[]
dd.append(heading)
import csv
'''with open('data.csv', mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    print(len(data_to_write))
    while row < len(data_to_write):
        col = 0
        line=""
        
        while col < 61:
            #if col < 9:
            #print(data_to_write[row][heading[col]])
            line+=' '
            line+=str(data_to_write[row][heading[col]])
            worksheet.write(row+1 , col ,data_to_write[row][heading[col]])
            #print(row)
            #print(col)
            col+=1
        data_writer.writerow(line)
        row+=1'''
while row < len(data_to_write):
    col = 0
    line=[]
    while col < 61:
        line.append(str(data_to_write[row][heading[col]]))
        #worksheet.write(row+1 , col ,data_to_write[row][heading[col]])
        col+=1
    row+=1
    dd.append(line)

with open('data.csv', mode='a', newline='') as data_file:
    writer = csv.writer(data_file)
    writer.writerows(dd)

#workbook.close()
print("ajhkqjekqje")

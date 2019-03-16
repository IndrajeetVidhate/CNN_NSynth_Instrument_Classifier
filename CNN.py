# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 17:03:53 2018

@author: vidha
"""

import torchvision.transforms as trans
import torch as ptorch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as ploty
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.tensor as tensor
import torch.utils.data as data
import torchvision.transforms as trans
import torchvision.transforms as transforms
from pytorch_nsynth.nsynth import NSynth
import numpy as np
import torch
import scipy.signal as converter
from scipy.io import wavfile
from scipy.fftpack import fft, dct
from scipy.signal import savgol_filter, stft
from scipy import signal
from numpy.lib import stride_tricks
import matplotlib.ticker as ticker



#import peakutils
#from pydub import AudioSegment

#from python_speech_features import mfcc

bs = 100
epochs =10

learningR = 0.01

# thing remaining Downsampling

#transformer = trans.Compose([ trans.Lambda(lambda x : toSpecto(x) ), trans.Lambda(lambda x : torch.Tensor(x) ), trans.Lambda(lambda x : x[0:64000:2]) ,trans.Lambda(lambda x : torch.unsqueeze(x,0))  ])
#transformer = trans.Compose([ trans.Lambda(lambda x : toSpecto(x) ), trans.Lambda(lambda x : torch.Tensor(x) ),trans.Lambda(lambda x : x.contiguous().view(1,-1)),   trans.Lambda(lambda x : torch.unsqueeze(x,0))  ])

#,trans.Lambda(lambda x : x.contiguous().view(1,-1))
transformer = trans.Compose([trans.Lambda(lambda x : torch.Tensor(x) ), trans.Lambda(lambda x : downsample(x) ) , trans.Lambda(lambda x : toSpecto(x) ), trans.Lambda(lambda x : torch.Tensor(x) ),trans.Lambda(lambda x : x.contiguous().view(1,-1)) ])


#transformer = trans.Compose([trans.Lambda(lambda x: x / np.iinfo(np.int16).max), trans.Lambda(lambda x : toSpecto(x) )  , trans.Lambda(lambda x : x[None, :, :] )])
#transformer = transforms.Compose([transforms.Lambda(lambda x: x / np.iinfo(np.int16).max), transforms.Lambda(lambda x: torch.from_numpy(x).float().unsqueeze(0))])


def downsample(x):
    
    
    x = x.unsqueeze(0)
    x = x.unsqueeze(0)
    x = F.interpolate(x, size = (48000), mode = 'linear')
    x = x .squeeze(0)
    x = x .squeeze(0)
    return x
    
def toSpecto(sample, fs = 16000):
   
     
     #f1,t1,z1 = signal.spectrogram(sample)
     f1,t1,z1 =  signal.stft(sample,16000)
   
     #z1 = np.asarray(z1, dtype=np.float)
     #sample_intermediate = samples.unsqueeze(0)    
     #samples1 = converterq(sample_intermediate)
     #converterq = nn.MaxPool1d(kernel_size=64, stride = 1)
     #z1 = 10*np.log10(np.abs(z1))
    
     
     
     return np.abs(z1)
     
#toFloat = transforms.Lambda(lambda x: x / np.iinfo(np.int16).max)

trainset = NSynth(
        "./nsynth-train",
        transform=transformer,
        blacklist_pattern=["synth_lead"],  # blacklist string instrument
        categorical_field_list=["instrument_family", "instrument_source"])
testset = NSynth(
        "./nsynth-test",
        transform=transformer,
        blacklist_pattern=["synth_lead"],  # blacklist string instrument
        categorical_field_list=["instrument_family", "instrument_source"])
validset = NSynth(
        "./nsynth-valid",
        transform = transformer,
        blacklist_pattern=["synth_lead"],  # blacklist string instrument
        categorical_field_list=["instrument_family", "instrument_source"])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers = 0, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=True, num_workers = 0, pin_memory=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=bs, shuffle=True, num_workers = 0, pin_memory=True)
  


class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
       
        
        self.conv1 = nn.Conv1d(1, 10, 5)
        self.conv2 = nn.Conv1d(10, 32, 5)
        self.conv3 = nn.Conv1d(32, 50, 3)
        self.conv4 = nn.Conv1d(50, 64, 3)
        self.conv5 = nn.Conv1d(64, 81, 3)
        self.conv6 = nn.Conv1d(81, 100, 3)
        self.conv7 = nn.Conv1d(100, 128, 3)
        self.batchNorm1 = nn.BatchNorm1d(10)
        self.batchNorm7 = nn.BatchNorm1d(70)
        self.batchNorm4 = nn.BatchNorm1d(64)
        self.fc1 = nn.Linear(48128, 1000)
        self.fc2 = nn.Linear( 1000, 156)
        self.fc3 = nn.Linear(156, 10)
        self.fcDropout = nn.Dropout(p = 0.4)
        self.Conv2Dropout = nn.Dropout(p = 0.1)

    def forward(self, x):
       
       
        x = F.max_pool1d(F.relu(self.batchNorm1(self.conv1(x))), 2,2)
       
        x =  F.max_pool1d(F.relu(self.conv2(x)), 2,2)
        x = F.max_pool1d(F.relu(self.conv3(x)), 2 ,2)
        x = F.max_pool1d(F.relu(self.batchNorm4(self.conv4(x))), 2, 2)
        x = F.max_pool1d(F.relu(self.conv5(x)), 2,2)
        x = F.max_pool1d(F.relu(self.conv6(x)), 2, 2)
        x = F.max_pool1d(F.relu(self.conv7(x)), 2, 2)
    
# =============================================================================
#         x =  F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = F.max_pool2d(F.relu(self.conv3(x)), 2)
#         x = F.max_pool2d(F.relu(self.conv4(x)), 2)
#         x = self.Conv2Dropout(F.max_pool2d(F.relu(self.conv5(x)), 2))
#         x = F.max_pool2d(F.relu(self.conv6(x)), 2)
#         x = F.max_pool2d(F.relu(self.batchNorm7(self.conv7(x))), 2)
# =============================================================================
        
        
# =============================================================================
#         x = F.max_pool1d(F.relu(self.conv3(x)), 2)
#         x = F.max_pool1d(F.relu(self.conv4(x)), 2)
#         x = F.max_pool1d(F.relu(self.conv5(x)), 2)
#         x = F.max_pool1d(F.relu(self.conv6(x)), 2)
#         x = F.max_pool1d(F.relu(self.batchNorm7(self.conv7(x))), 2)
# =============================================================================
        
# =============================================================================
#         x = F.max_pool1d(F.relu(self.conv3(x)), 2)
#         x = F.max_pool1d(F.relu(self.conv4(x)), 2)
#         x = F.max_pool1d(F.relu(self.conv5(x)), 2)
#         x = F.max_pool1d(F.relu(self.conv6(x)), 2)
#         x = F.max_pool1d(F.relu(self.conv7(x)), 2)
#         
# =============================================================================
        x = x.view(-1, self.num_flat_features(x))
        
        x = self.fcDropout(F.relu(self.fc1(x)))
        x = self.fcDropout(F.relu(self.fcDropout(self.fc2(x))))
        x = self.fcDropout(self.fc3(x))
        return F.log_softmax(x)
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    
    
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv_layer_1 = nn.Conv2d(1,6,5)
        self.conv_layer_2 = nn.Conv2d(6,16,5)
        self.conv_layer_3 = nn.Conv2d(16,56,5)
        self.conv_layer_4 = nn.Conv2d(16,32,5)
        self.conv_layer_5 = nn.Conv2d(128,256,5)
        self.conv_layer_6 = nn.Conv2d(256,512,5)
        
             
        self.function_1 = nn.Linear(32,114688)
        self.function_2 = nn.Linear(114688,64000)
        self.function_3 = nn.Linear(64000,10)
        

    def forward(self, x):
        
       
        
        x = F.max_pool2d(F.relu(self.conv_layer_1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv_layer_2(x)),2)
        x = F.max_pool2d(F.relu(self.conv_layer_3(x)),2)
        
# =============================================================================
#         x = F.max_pool2d(F.relu(self.conv_layer_4(x)),2
#         x = F.max_pool2d(F.relu(self.conv_layer_4(x)),2)
#         x = F.max_pool2d(F.relu(self.conv_layer_5(x)),2)
#         x = F.max_pool2d(F.relu(self.conv_layer_6(x)),2)
#         x = F.max_pool2d(F.relu(self.conv_layer_7(x)),2)
# =============================================================================
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.function_1(x))
        x = F.relu(self.function_2(x))
        x = self.function_3(x)
        x= F.softmax(x)
        return x
        
    def num_flat_features(self, x):
      size = x.size()[1:]
      num_features = 1
      for s in size:
          num_features *= s 
      return num_features

    

def train():
    
    
    network2 = Net1()
    network2 = network2.train()
    network2= network2.cuda()
    #optimizer = torch.optim.Adam(network2.parameters(), lr = learningR)
    optimizer = torch.optim.SGD(network2.parameters(), lr= learningR, momentum=0.8)
    loss = nn.CrossEntropyLoss();
    loss_array_train = []
    loss_array_validation = []
    epoch_array = []
    currentEpoch = 0
    combinedCount = 0
    for epoch in range(epochs):
        currentEpoch+=1
        per_complete_epoch_loss = 0
        
        total_batches = 1
        total_batches_valid = 1
        per_complete_epoch_loss_valid = 0
        epoch_array.append(epoch)
        batch_no = 0
        print(str(epoch) + '.... going on ')
        
        for samples, instrument_family_target, instrument_source_target, targets in trainloader:
            
            
            
   
            ptorch.cuda.empty_cache()
            batch_no+=1 
            optimizer.zero_grad()
            
            samples = samples.cuda()
            instrument_family_target = instrument_family_target.cuda()
            samples = Variable(samples)
            out = network2(samples)
            
            losst = loss(out, instrument_family_target)
            losst.backward()
            torch.nn.utils.clip_grad_norm_(network2.parameters(), 6)
            optimizer.step()
            per_complete_epoch_loss+= losst.data.item()
            total_batches+=1
            
            print('training loss ' + str(losst.data.item()))
            print(str(epoch) + '.... going on ')
            del losst
            del out
            
        
        for samples, instrument_family_target, instrument_source_target, targets in validloader:
            instrument_family_target = instrument_family_target.cuda()
            samples = samples.cuda()
            samples = Variable(samples)
            valid_out = network2(samples)
            lossv = loss(valid_out,instrument_family_target)
            per_complete_epoch_loss_valid += lossv.data.item()
            total_batches_valid+=1
            print('validation loss ' + str(lossv.data.item()))
            del lossv
            del valid_out
            
        print(' Training Loss for epoch ' + str(epoch) +" : "+ str(per_complete_epoch_loss/total_batches))
        print(' Valid Loss for epoch ' + str(epoch) +" :"+ str(per_complete_epoch_loss_valid/total_batches_valid))
        loss_array_validation.append(per_complete_epoch_loss_valid/total_batches_valid)    
        loss_array_train.append(per_complete_epoch_loss/total_batches)
          
        
    ptorch.save(network2.state_dict(), "./CNN_1.pt")
    plot(loss_array_train, loss_array_validation, currentEpoch)
    return network2
# =============================================================================
#             Spec,f,b,t = ploty.specgram(samples[0], NFFT = 1024, Fs=16000, noverlap = 900)
#             Spec = 10*np.log10(Spec)
#             ploty.imshow(Spec)
# =============================================================================

def transferred():
    ptorch.cuda.empty_cache()
    network2 = Net1()
    state_dict = ptorch.load("./73_Best.pt")
    network2.load_state_dict(state_dict)
    network2 = network2.cuda()
    return network2
    
def plot(saved_losses, saved_losses_valid, currentEpoch):
    fig, ax = ploty.subplots()
    
    x = np.linspace(1, currentEpoch, currentEpoch)
    saved_losses = np.array(saved_losses)
    slv = np.array(saved_losses_valid)

    ax.set_title("Average Model Loss over Epochs")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average Loss")
    
    #Adjust x-axis ticks
    tick_spacing = 5
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    
    ax.plot(x, saved_losses, color='purple', marker=".")
    ax.plot(x, slv, color='red', marker=".")
    fig.savefig('Network2_Loss_Epoch')
    


def imageDraw(z1, name):
    
    
    to_plot = z1.view(-1,172)
    
    ploty.imsave(fname = name + "class.png", arr = to_plot )
# =============================================================================
# 	 ploty.imsave(fname = 'with_high_'+name + '.png', arr = )
# 	 ploty.title('STFT Magnitude')
# 	 ploty.ylabel('Frequency [Hz]')
# 	 ploty.xlabel('Time [sec]')
# 	 ploty.savefig('with_high_'+name + '.png')
# =============================================================================
    
    
def test(network2):
    
    network2 = network2.eval()
    correct_predictions = 0
    total_count = 0
    class_correct = list(0. for i in range (10))
    class_total = list(1. for i in range (10))
    classes = list(i for i in range (10))
    histo_x = []
    accuracies = list(0. for i in range (10))
    histo_labels = ['0','1','2','3','4','5','6','7','8','9']
    cm = [[0. for i in range(10)] for j in range(10)]
    trial = []
    count_high = 0
    with ptorch.no_grad():    
       
        for samples, instrument_family_target, instrument_source_target, targets in testloader:
            
            
        
                        
            samples = samples.cuda()
            instrument_family_target = instrument_family_target.cuda()
            out = network2(samples)
            _, pred_y = ptorch.max(out.data,1)
            total_count += samples.data.size()[0]
            correct_predictions +=  (pred_y == instrument_family_target).sum()
            
            
# =============================================================================
#             if(count_high < 11):
#                 for i in range (10):
#                     for j in range (len (instrument_family_target)):
#                         if (instrument_family_target[j] == pred_y[j] == i):
#                             imageDraw( samples[j][0], str(i) + "_sample with high" )
#                             count_high+=1
#             
# =============================================================================
            if(count_high < 11):
                for i in range (10):
                    for j in range (len (instrument_family_target)):
                        if (instrument_family_target[j] != pred_y[j] and instrument_family_target[j] == i):
                            trial = out[j].cpu().numpy()
                            trial = np.exp(trial)
                            #trial = np.sort(trial)
                            if (abs(trial[pred_y[j]] - trial[instrument_family_target[j]]) < 0.20):
                                imageDraw( samples[j][0], str(i) + "_sample_margin_incorrect" )
                                count_high+=1
            
                            
                        
            
            for i in range(len(instrument_family_target)):
                cm[instrument_family_target[i]][pred_y[i]]+=1
            
            for i in range (len(instrument_family_target)):
                
                
                    if ( instrument_family_target[i] == pred_y[i]):
                        class_correct[instrument_family_target[i]]+=1
                        class_total[instrument_family_target[i]]+=1
                        #histo_x.append(y[i])
                        
                    else:
                        #class_correct[pred_y[i]] +=1
                        class_total[instrument_family_target[i]]+=1
                       
         
                       
        print("class wise accuracy")
        for i in range (10):
            accuracies[i] = (class_correct[i] / class_total[i])
            print('Accuracy of %5s : %2f %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
        print(accuracies)
        print( "Complete accuracy :" + str ((correct_predictions.item()/total_count)*100))
        print("Confusion matrix")
        for i in cm:
            print(i)
            
            
    
    
if __name__=='__main__':
# =============================================================================
# 
#     t = torch.zeros([1,4], dtype=torch.float32)
#     t[0][0] = 5.0
#     t[0][1] = 10.0
#     t[0][2] = 15.0
#     t[0][3] = 20.0
#     print(t)
#     t = t.unsqueeze(0)
#     print("1 squeeze")
#     print(t)
#     t = F.interpolate(t, size = (2), mode = 'linear')
#     t = t.squeeze(0)
#     print(t)
# =============================================================================
    #network2 = train()
    network2 = transferred()
    test(network2)
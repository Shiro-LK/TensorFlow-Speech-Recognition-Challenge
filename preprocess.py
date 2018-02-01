# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 15:25:04 2017

@author: shiro
"""
import acoustics
from scipy import signal
import math
import random
import numpy as np
from collections import Counter
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa
import librosa.display
import webrtcvad

def prepare_data(filename):
    '''
        load file which give path and label for the data
    '''
    f = open(filename, 'r')
    
    data = [line.split() for line in f]
    feat =[]
    label=[]
    for l in data:
        feat.append(l[0])
        label.append(l[1])
   
    count = Counter(label)
    print(count)
    label = np.array(label, dtype=np.int)
    return feat, label

def get_raw(paths, nsamples=16000, resample=False, nresample=8000, data_aug=False, proba=0.5, coeff_amplitude=False, coeff_time=4000):
    '''
        Given list of paths, return raw data
        nsample = number of samples per second.
    '''
    T = len(paths)
    #print('Size : ', T)
    # read the wav files
    wavs = [wavfile.read(x)[1] for x in paths]
    
    # zero pad the shorter samples and cut off the long ones.
    data = [] 
    
    for wav in wavs:
        if wav.size < 16000:
            d = np.pad(wav, (nsamples - wav.size, 0), mode='constant')
        else:
            d = wav[0:nsamples]
            
        
        if data_aug == True:
            #print('true')
            d = data_augmentation(d, proba=proba, coeff_amplitude=coeff_amplitude, coeff_time=coeff_time)
            
        if resample == True:
            d = signal.resample(d, nresample)
        data.append(d)


       
    return np.asarray(data).reshape(T, -1, 1) #format conv 1D

def getPower(clip):
  clip2 = clip.copy()
  clip2 = np.array(clip2) / (2.0**15)  # normalizing
  clip2 = clip2 **2
  return np.sum(clip2) / (len(clip2) * 1.0)


def addNoise(audio, ampl = False):
     snrTarget = np.random.randint(15,30)
     c = ['brown', 'blue', 'violet', 'white', 'pink']
     color = np.random.choice(c, 1)[0]
     noise = np.array(((acoustics.generator.noise(audio.shape[0], color=color))/3) * 32767).astype(np.int64)
     sigPower = getPower(audio)
     noisePower = getPower(noise) 
     factor = (sigPower / noisePower ) / (10**(snrTarget / 20.0))  # noise Coefficient for target SNR
     coeff = 1.0
     if ampl == True:        
        coeff = np.random.uniform(0.9, 1.1)
     return np.int16( (audio + noise * np.sqrt(factor))* coeff)

def add_noise(data, coeff=0.5):
    c = ['brown', 'blue', 'violet', 'white', 'pink']
    color = np.random.choice(c, 1)[0]
    val = np.random.uniform(0.0, coeff)
    noise = np.array(((acoustics.generator.noise(data.shape[0], color=color))/3) * 32767/100).astype(np.int64)
    # equilibrate energy
    noise_energy = np.sqrt(np.sum(noise**2) / noise.size)
    
    data_energy = np.sqrt(np.sum(data**2) / data.size)
#    print(data_energy/noise_energy)
    print(val * noise * data_energy / noise_energy)
#    print(data)
    return np.int16(data + val * noise * data_energy / noise_energy)

def time_shift(data, time_max=5000):
    newdata = np.zeros(data.shape, dtype=np.int16)
    begin = np.random.randint(-time_max,time_max)# time_max exclut
    if begin>=0:
        newdata[begin:] = data[0:data.shape[0]-begin]
    else:
        T = data.shape[0]
        newdata[:T-abs(begin)] = data[abs(begin):]
    return newdata
    
def data_augmentation2(data, proba=0.5, coeff_noise=0.25, coeff_time=1000):
    if np.random.uniform(0.0, 1.0) <= proba:
        print('aug')
        data = time_shift(data, time_max=coeff_time)
        data = add_noise(data, coeff_noise)
        
    return data#np.int16(data/np.max(np.abs(data)) * 32767)

def data_augmentation(data, proba=0.5, coeff_amplitude=False, coeff_time=4000):
    if np.random.uniform(0.0, 1.0) <= proba:
        data = time_shift(data, time_max=coeff_time)
        data = addNoise(data, coeff_amplitude)
        #print(data.dtype)
    return data

def log_spectrogram(path, nsamples=16000, window_size=20, step_size=10, eps=1e-10, data_aug=False, proba_data_aug=0.5, coeff_amplitude=False, coeff_time=4000):
    '''
        Given path, return specgram.
    '''
    # read the wav files
    sample_rate, wav = wavfile.read(path) # 16000 samples per second
    
    # zero pad the shorter samples and cut off the long ones to have a signal of 1 sec.
    if wav.size < nsamples:
        d = np.pad(wav, (nsamples - wav.size, 0), mode='constant')
    else:
        d = wav[0:nsamples]
    
    if data_aug == True:
        d = data_augmentation(d, proba=proba_data_aug, coeff_amplitude=coeff_amplitude, coeff_time=coeff_time)
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    # get the specgram
    freq, times, specgram = signal.spectrogram(d, fs= nsamples ,window='hann', nperseg=nperseg, noverlap=noverlap, detrend=False)
    
    return freq, times, np.log(specgram.T.astype(np.float32) + eps)

def log_mel_spectrogram(path, n_mels=40, nsamples=16000, data_aug=False, proba_data_aug=0.5, coeff_amplitude=False,
                    coeff_time=4000):
    '''
    Given path, return mel's coef.
    '''
    # read the wav files
    sample_rate, sample = wavfile.read(path)  # 16000 samples per second

    # zero pad the shorter samples and cut off the long ones to have a signal of 1 sec.

    if sample.size < nsamples:
        d = np.pad(sample, (nsamples - sample.size, 0), mode='constant')
    else:
        d = sample[0:sample_rate]

    if data_aug:
        d = data_augmentation(d, proba=proba_data_aug, coeff_amplitude=coeff_amplitude, coeff_time=coeff_time)


    # get the coefs
    S = librosa.feature.melspectrogram(d, sr=nsamples, n_mels=n_mels)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.

    log_S = librosa.power_to_db(S, ref=np.max)

    return log_S

def fast_log_mel_spectrogram(path, n_mels=40, nsamples=16000, data_aug=False, proba_data_aug=0.5, coeff_amplitude=False,
                    coeff_time=4000, cut_accuracy=40, new_sample_rate = 16000):
    '''
    Given path, return mel's coef.
    '''
    # read the wav files
    sample_rate, sample = wavfile.read(path)  # 16000 samples per second

    vad = webrtcvad.Vad() #We use VAD to remove silences
    vad.set_mode(1)

    voice = False
    start=0
    end=cut_accuracy
    while not voice:
        frame = sample[start:end]
        voice = vad.is_speech(frame, sample_rate)
        start = end+1
        end += cut_accuracy
    sample=sample[start:]

    if sample_rate != new_sample_rate:
        sample = signal.resample(sample, int(new_sample_rate / sample_rate * sample.shape[0]))
        sample_rate=new_sample_rate

    # zero pad the shorter samples and cut off the long ones to have a signal of 1 sec.
    if sample.size < nsamples:
        d = np.pad(sample, (nsamples - sample.size, 0), mode='constant')
    else:
        d = sample[0:sample_rate]

    if data_aug:
        d = data_augmentation(d, proba=proba_data_aug, coeff_amplitude=coeff_amplitude, coeff_time=coeff_time)


    # get the coefs
    S = librosa.feature.melspectrogram(d, sr=nsamples, n_mels=n_mels)

    # Convert to log scale (dB). We'll use the peak power (max) as reference.

    log_S = librosa.power_to_db(S, ref=np.max)

    return log_S

    
def load_data_with_spectrogram(paths, nsamples=16000, transpose=False, p='', window_size=20, step_size=10, eps=1e-10, data_aug=False, proba_data_aug=0.5, coeff_amplitude=False, coeff_time=4000):
    '''
        return array of spectrogram (number, times, feat) by default from list of paths
    '''
    #print(data_aug)
    if transpose==False:
        data = np.asarray([ log_spectrogram(p+path, nsamples=nsamples, window_size=window_size, step_size=step_size, eps=eps, data_aug=data_aug,  proba_data_aug=proba_data_aug, coeff_amplitude=coeff_amplitude, coeff_time=coeff_time)[2] for path in paths]) # freq, windows
    else:
        data = np.asarray([ log_spectrogram(p+path, nsamples=nsamples, window_size=window_size, step_size=step_size, eps=eps, data_aug=data_aug,  proba_data_aug=proba_data_aug, coeff_amplitude=coeff_amplitude, coeff_time=coeff_time)[2].T for path in paths]) # windows, freq
    #print('data : ', data.shape)
    return data

def load_data_with_mel_spectrogram(paths, transpose=False, p='', nsamples=16000, n_mels=40, data_aug=False,
                               proba_data_aug=0.5, coeff_amplitude=False, coeff_time=4000, fast=False, new_sample_rate = 16000):
    '''
        return array of melspectrogram (number, n_mel, times) by default from list of paths
    '''
    # print(data_aug)
    if fast:
        if transpose:
            data = np.asarray([fast_log_mel_spectrogram(p + path, n_mels=n_mels, nsamples=nsamples,
                                               data_aug=data_aug, proba_data_aug=proba_data_aug, coeff_amplitude=coeff_amplitude,
                                               coeff_time=coeff_time,new_sample_rate = new_sample_rate).T for path in paths])  # freq, windows
        else:
            data = np.asarray([fast_log_mel_spectrogram(p + path, n_mels=n_mels, nsamples=nsamples,
                                               data_aug=data_aug, proba_data_aug=proba_data_aug, coeff_amplitude=coeff_amplitude,
                                               coeff_time=coeff_time,new_sample_rate = new_sample_rate) for path in paths])  # windows, freq
    else:
        if transpose:
            data = np.asarray([log_mel_spectrogram(p + path, n_mels=n_mels, nsamples=nsamples,
                                               data_aug=data_aug, proba_data_aug=proba_data_aug, coeff_amplitude=coeff_amplitude,
                                               coeff_time=coeff_time).T for path in paths])  # freq, windows
        else:
            data = np.asarray([log_mel_spectrogram(p + path, n_mels=n_mels, nsamples=nsamples,
                                               data_aug=data_aug, proba_data_aug=proba_data_aug, coeff_amplitude=coeff_amplitude,
                                               coeff_time=coeff_time) for path in paths])  # windows, freq
    # print('data : ', data.shape)
    return data
 
def plot_spectrogram(freqs, times, spec):
    ax2 = plt.imshow(spec.T , aspect='auto', origin='lower', extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    #ax2.set_ylabel('Freqs in Hz')
    #ax2.set_xlabel('Seconds')
def main():
    file = 'train/audio/bed/00176480_nohash_0.wav'
    plt.figure(1)
    plt.subplot(211)
    
    freqs, times, spec = log_spectrogram(file, data_aug=True, coeff_noise=0.25, coeff_time=4000)
    plot_spectrogram(freqs, times, spec)
    
    plt.subplot(212)
    freqs, times, spec = log_spectrogram(file, data_aug=False)
    plot_spectrogram(freqs, times, spec)
    print(spec.shape) #(99,161)

def main4():
    file = 'train/audio/bed/00176480_nohash_0.wav'
    sample_rate, wav = wavfile.read(file) # 16000 samples per second
    print(wav)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(wav)
    
    wav_noise = get_raw([file], data_aug=True, proba=0.7, coeff_time=4000)
    wav_noise = wav_noise.reshape((-1))
    print(wav_noise.dtype)
    plt.subplot(212)
    plt.plot(wav_noise)
    plt.show()
    
    plt.subplot(311)
    plt.plot(signal.wiener(wav_noise))
    plt.show()
def main5():
    file = ['train/audio/bed/00176480_nohash_0.wav']
    data = load_data_with_mel_spectrogram(file, False)
def main6():
    file = ['train/audio/bed/00176480_nohash_0.wav']
    data = load_data_with_mel_spectrogram(paths=file, transpose=True, n_mels=40, data_aug=False)
    print(data.shape)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(data[0], sr=16000, x_axis='time', y_axis='mel')
    plt.title('Mel power spectrogram ')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()

    
#main6()
#Réduire la fréquence à 8000Hz
#Réduire la taille des sons en commencant d'une belle manière
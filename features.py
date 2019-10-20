import pywt
import numpy as np
from scipy import signal,fftpack
from scipy.io import wavfile
import matplotlib.pyplot as plt

def lowPassFilter(sig, f_max, fs, num_taps):
    lowPass = signal.firwin(num_taps, cutoff = 2*f_max/fs,window = 'hamming') #cutoff frequency in terms of nyquist rate
    return np.convolve(sig, lowPass, mode = 'same'), lowPass

#Guitar
fs,audio = wavfile.read('guitarslow.wav')
time = np.arange(0,len(audio)/fs,1/fs)
audio = np.asarray(audio)
# audio = audio[:,1]
plt.plot(time, audio)
plt.title('Guitar')
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.show()

print(len(audio))

audio = audio[:55000]
time = time[:55000]
#Envelope detection
audio_abs = abs(audio)
print(len(audio_abs))
audio_abs = np.split(audio_abs, 500)
# print(audio)
envelope = []
for i in audio_abs :
    max = np.amax(i)
    # print(max)
    for j in i :
        envelope.append(max)

y,h = lowPassFilter(envelope, 20, fs, 51)

plt.plot(time, y)
plt.title('Guitar Envelope')
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.show()
F = fftpack.fft(envelope)
freq = fftpack.fftfreq(len(envelope))*fs
plt.plot(freq,abs(F))
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.title('Fourier Transform')
# plt.xlim(0,1000)
plt.show()

#Flute
fs,audio = wavfile.read('FluteClean2_A2.wav')
time = np.arange(0,len(audio)/fs,1/fs)
audio = np.asarray(audio)
# audio = audio[:,1]
plt.plot(time, audio)
plt.title('Flute')
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.show()

print(len(audio))

audio = audio[:10000]
time = time[:10000]
#Envelope detection
audio_abs = abs(audio)
print(len(audio_abs))
audio_abs = np.split(audio_abs, 100)
# print(audio)
envelope = []
for i in audio_abs :
    max = np.amax(i)
    # print(max)
    for j in i :
        envelope.append(max)

plt.plot(time, envelope)
plt.title('Flute Envelope')
plt.xlabel('time')
plt.ylabel('Amplitude')
plt.show()
F = fftpack.fft(envelope)
freq = fftpack.fftfreq(len(envelope))*fs
plt.plot(freq,abs(F))
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.title('Fourier Transform')
# plt.xlim(0,1000)
plt.show()

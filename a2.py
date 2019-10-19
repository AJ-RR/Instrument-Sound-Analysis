import pywt
import numpy as np
from scipy import signal,fftpack
from scipy.io import wavfile
import matplotlib.pyplot as plt
fs,audio = wavfile.read('guitarsuper.wav')
time = np.arange(0,len(audio)/fs,1/fs)
audio = np.asarray(audio)
audio = audio[:,1]
plt.plot(time, audio)
plt.show()
print(len(audio))
# print(len(audio[1]),fs)
#spectrum of audio signal
F = fftpack.fft(audio)
freq = fftpack.fftfreq(len(audio))*fs
plt.plot(freq,abs(F))
plt.title('Fourier Transform')
plt.xlim(0,1000)
plt.show()

#spectrogram
f, t , Sxx = signal.spectrogram(audio,fs, nfft = 8192)
plt.pcolormesh(t, f, Sxx)
plt.yscale("symlog")
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (second)')

plt.show()

# #Spectrogram of the signal
# plt.specgram(audio, Fs = fs, NFFT = 8192)
# plt.xlabel('time')
# plt.ylabel('frequency')
# plt.ylim(0,1000)
# plt.show()

#Scalogram of the signal
widths = np.arange(1,21)
cwt_matrix, freqs = pywt.cwt(audio, widths, 'morl')
plt.imshow(cwt_matrix, extent=[-1, 1, 1, 21], cmap='PRGn', aspect='auto', vmax=abs(cwt_matrix).max(), vmin=-abs(cwt_matrix).max())
plt.title('scalogram')
plt.show()

# #Wavelet Transform of the signal
# widths = np.arange(1,21)
# cwt_matrix = signal.cwt(audio, signal.gabor, widths)
#
# plt.imshow(cwt_matrix, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto')
# plt.show()

# fs,audio = wavfile.read('flutefull.wav')
# time = np.arange(0,len(audio)/fs,1/fs)
# audio = np.asarray(audio)
# audio = audio[:,1]
# plt.plot(time, audio)
# plt.show()
# print(len(audio))
# # print(len(audio[1]),fs)
# #spectrum of audio signal
# F = fftpack.fft(audio)
# freq = fftpack.fftfreq(len(audio))*fs
# plt.plot(freq,abs(F))
# plt.title('Fourier Transform')
# plt.xlim(0,1000)
# plt.show()
#
# #spectrogram
# f, t , Sxx = signal.spectrogram(audio,fs, nfft = 4096)
# plt.pcolormesh(t, f, Sxx)
# plt.yscale("symlog")
# plt.ylabel('Frequency (Hz)')
# plt.xlabel('Time (second)')
#
# plt.show()

# #Spectrogram of the signal
# plt.specgram(audio, Fs = fs, NFFT = 8192)
# plt.xlabel('time')
# plt.ylabel('frequency')
# plt.ylim(0,2500)
# plt.show()

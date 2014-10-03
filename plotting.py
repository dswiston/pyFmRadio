from matplotlib import pyplot as plot
import matplotlib

plot.xkcd();
prop = matplotlib.font_manager.FontProperties(fname=matplotlib.font_manager.path+'/Humor-Sans.ttf', size=16)

# Plot the entire 1 second of audio
numAudioSamps = np.size(audioDataLplusR);
plot.plot(np.linspace(0,numAudioSamps/40000,numAudioSamps),audioDataLplusR); plot.title('Audio Signal');
plot.xlabel('Time (sec)'); plot.ylabel('Amplitude');
plot.show(); 


# Plot the entire spectrum for the 1 second collection
win = np.hamming(np.size(data));
freqSpec = 20*np.log10(np.abs(np.fft.fft(data*win)));
plot.plot(np.linspace(90,90.2,400000), freqSpec[0:400000]); plot.title('Frequency Spectrum');
plot.xlabel('Frequency (MHz)'); plot.ylabel('Amplitude (dBx)'); plot.ylim(20,100);
plot.annotate('Radio Station @ 90.1 FM',xy=(90.1,88), arrowprops=dict(arrowstyle='->'), xytext=(90.01,90))
plot.annotate('HD Radio Station OFDM',xy=(90.04,53), arrowprops=dict(arrowstyle='->'), xytext=(90.065,30))
plot.annotate('HD Radio Station OFDM',xy=(90.165,53), arrowprops=dict(arrowstyle='->'), xytext=(90.065,30))
plot.show(); 


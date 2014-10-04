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
plot.plot(np.linspace(89.9,90.3,400000), freqSpec[0:400000]); plot.title('Frequency Spectrum');
plot.xlabel('Frequency (MHz)'); plot.ylabel('Amplitude (dBx)'); plot.ylim(0,80);
plot.annotate('Radio Station @ 90.1 FM',xy=(90.1,61), arrowprops=dict(arrowstyle='->'), xytext=(90.01,70))
plot.annotate('HD Radio Station OFDM',xy=(90.23,30), arrowprops=dict(arrowstyle='->'), xytext=(90.025,9))
plot.annotate('HD Radio Station OFDM',xy=(89.98,30), arrowprops=dict(arrowstyle='->'), xytext=(90.025,9))
plot.show(); 



win = np.hamming(np.size(fmDemod));
freqSpec = 20*np.log10(np.abs(np.fft.fft(fmDemod*win)));
plot.plot(np.linspace(-100,100,204800), np.fft.fftshift(freqSpec)); plot.title('Frequency Spectrum');
plot.xlabel('Frequency (KHz)'); plot.ylabel('Amplitude (dBx)'); plot.ylim(20,100);
plot.annotate('L+R Audio Signal',xy=(0,85), arrowprops=dict(arrowstyle='->'), xytext=(-25,95))

plot.annotate('L-R Audio Signal',xy=(31,53), arrowprops=dict(arrowstyle='->'), xytext=(-25,30))
plot.annotate('L-R Audio Signal',xy=(-31,53), arrowprops=dict(arrowstyle='->'), xytext=(-25,30))

plot.annotate('19KHz Pilot',     xy=(-17,61.75), arrowprops=dict(arrowstyle='->'), xytext=(-19,50))
plot.annotate('19KHz Pilot',     xy=(17,61.75), arrowprops=dict(arrowstyle='->'), xytext=(-19,50))

plot.show(); 




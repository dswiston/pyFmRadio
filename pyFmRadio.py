#!/usr/bin/python

import threading
import Queue
import time
import struct
import ao
import numpy as np
from scipy.signal import lfilter
from scipy.signal import hilbert
from rtlsdr import RtlSdr


class FileReader(threading.Thread):
  def run(self):

    fid = open('/home/dave/testing3','rb');
    blkSize = 1024*1000;
    data = fid.read(blkSize);
    while len(data) >= 1:
      data = ConvertData(data,len(data));
      dataQueue.put(data);
      dataQueue.join();
      data = fid.read(blkSize);
            

def sdrCallback(samples, rtlsdr_obj):
  dataQueue.put(np.array(samples));
    

class FMDemod(threading.Thread):
  def run(self):

    time.sleep(1);
    
    # Create filter to downsample to 200KHz
    stageOneFilt = [-3.59558849352512e-05,-8.96980379207209e-05,-0.000177018129702616,-0.000284225463855195,-0.000377179255733887,-0.000392842855176856,-0.000238992154324545,0.000197162108922646,0.00102950812178301,0.00234332733360650,0.00415920831004696,0.00639739693554639,0.00885235663103450,0.0111873998115123,0.0129580383890179,0.0136678758676256,0.0128546099437662,0.0101957113051566,0.00561640193005583,-0.000622297788257071,-0.00787842641679388,-0.0151456025343175,-0.0211409442597648,-0.0244612258199640,-0.0237900539217606,-0.0181259503597799,-0.00699298796011055,0.00940618872114674,0.0301288876898665,0.0535430328936014,0.0774918667314895,0.0995541183265144,0.117362670555239,0.128932259393025,0.132942985095165,0.128932259393025,0.117362670555239,0.0995541183265144,0.0774918667314895,0.0535430328936014,0.0301288876898665,0.00940618872114674,-0.00699298796011055,-0.0181259503597799,-0.0237900539217606,-0.0244612258199640,-0.0211409442597648,-0.0151456025343175,-0.00787842641679388,-0.000622297788257071,0.00561640193005583,0.0101957113051566,0.0128546099437662,0.0136678758676256,0.0129580383890179,0.0111873998115123,0.00885235663103450,0.00639739693554639,0.00415920831004696,0.00234332733360650,0.00102950812178301,0.000197162108922646,-0.000238992154324545,-0.000392842855176856,-0.000377179255733887,-0.000284225463855195,-0.000177018129702616,-8.96980379207209e-05,-3.59558849352512e-05,0]
    # Create filter to downsample to 20KHz
    stageTwoFilt = [-0.00119626911550306,-0.00142546170364882,-0.00187744502104585,-0.00201701676228193,-0.00162817459020468,-0.000557487335421441,0.00122343806080337,0.00356350639674811,0.00612689813101657,0.00842445178475785,0.00990383156440624,0.0100757692616582,0.00865198056849580,0.00566181492072065,0.00150809535459767,-0.00305861160383326,-0.00706234834705832,-0.00950392490765292,-0.00960435635798143,-0.00703550797431447,-0.00207570414372595,0.00435841720332892,0.0108345011188760,0.0156744184636397,0.0173275806253855,0.0147733535863384,0.00786009225733559,-0.00250576384548369,-0.0143854332021002,-0.0250689244534925,-0.0315407793252962,-0.0310705762181423,-0.0218119645129463,-0.00328372400193613,0.0233795369353430,0.0554763088057064,0.0890971156152812,0.119739200111151,0.143065983583302,0.155666796524906,0.155666796524906,0.143065983583302,0.119739200111151,0.0890971156152812,0.0554763088057064,0.0233795369353430,-0.00328372400193613,-0.0218119645129463,-0.0310705762181423,-0.0315407793252962,-0.0250689244534925,-0.0143854332021002,-0.00250576384548369,0.00786009225733559,0.0147733535863384,0.0173275806253855,0.0156744184636397,0.0108345011188760,0.00435841720332892,-0.00207570414372595,-0.00703550797431447,-0.00960435635798143,-0.00950392490765292,-0.00706234834705832,-0.00305861160383326,0.00150809535459767,0.00566181492072065,0.00865198056849580,0.0100757692616582,0.00990383156440624,0.00842445178475785,0.00612689813101657,0.00356350639674811,0.00122343806080337,-0.000557487335421441,-0.00162817459020468,-0.00201701676228193,-0.00187744502104585,-0.00142546170364882,-0.00119626911550306]
    # Define the decimation rates at each stage        
    stageOneDec = 5;
    stageTwoDec = 5;         
    
    # Get the next chunk of data
    data = dataQueue.get();
    dataQueue.task_done();
    blkSize = len(data);
    
    # Downmix the signal, the dongle has a DC filter so we dwell offset
    demodFreq = 200e3;
    # FM carrier frequency
    carrierFreq = 19e3;
    # Define the raw sampling rate
    fs = 1e6;
    # Create the demodulation signal to baseband the FM station
    sigTime = np.linspace(0,blkSize,blkSize);
    sigTime = sigTime / fs;
    demodSig = np.exp(-1j*2*np.pi*demodFreq*sigTime);
    # Create the IIR peaking filter that isolates the carrier
    # This is important for demodulating the DSB-CS R-L signal that is used for stereo FM
    filtFreq = carrierFreq / demodFreq * 2;
    bw = 5 / demodFreq;
    [carrierFiltB,carrierFiltA] = PeakFilterDesign(filtFreq,bw);
       
    while(1):
                    
      # Keep only 200KHz of BW
      decData = PolyphaseDecimate(stageOneFilt,data,demodSig,0,stageOneDec);
      # Perform the FM Demodulation step
      fmDemod = FmDemodulate(decData);
      # Add an element to keep a proper size for the polyphase filtering
      fmDemod = np.concatenate(([0],fmDemod));
  
      # Filter the input to the audio rate (20KHz)
      audioDataLplusR = np.real(PolyphaseDecimate(stageTwoFilt,fmDemod,[],0,stageTwoDec));
      
      # Isolate the carrier signal
      stereoCarrier = lfilter(carrierFiltB,carrierFiltA,fmDemod);
      # Scale the carrier to full-scale since it will be used to demodulate
      stereoCarrier = stereoCarrier / np.sqrt(np.mean(stereoCarrier**2)) / np.sqrt(2);
      # Convert the carrier to a complex representation, then double the frequency
      # This signal is used demodulate the L-R DSB-SC signal 
      stereoDemod = hilbert(stereoCarrier)**2;
      # Filter the input to extract the L-R signal 
      audioDataLminusR = np.imag(PolyphaseDecimate(stageTwoFilt,fmDemod,stereoDemod,0,stageTwoDec));
      
      # Scale and DC filter the audio before playing it
      (audioDataLplusR, audioDataLminusR) = ProcessAudio(audioDataLplusR,audioDataLminusR);

      # Apply the de-emphasis filter
      audioFs = fs/stageOneDec/stageTwoDec;
      audioDataLplusR = DeEmphasisFilter(audioDataLplusR,audioFs);
      audioDataLminusR = DeEmphasisFilter(audioDataLminusR,audioFs);

      # Using the two audio signals (L+R) and (R-L), create the stereo stereo channels
      audioDataL = audioDataLplusR + audioDataLminusR;
      audioDataR = audioDataLplusR - audioDataLminusR;

      # Multiplex the stereo channels, the ao library assumes this behavior
      # Also convert the data to 16bit signed as the audio stream assumes this format
      audioData = np.empty(len(audioDataL)*2, dtype=np.int16);
      audioData[::2] = audioDataL.astype(np.int16);
      audioData[1::2] = audioDataR.astype(np.int16);

      # If mono is desired, uncomment this line
      #audioData = audioDataLplusR;

      # Put the audio onto the queue and wait for it to be received
      audioQueue.put(audioData.astype(np.uint16));
      audioQueue.join();
      
      # Get the next chunk of data
      data = dataQueue.get();
      dataQueue.task_done();
            

class AudioPlay(threading.Thread):
  def run(self):

    # Create the audio device - change channels to '1' if mono is desired
    pcm = ao.AudioDevice("pulse", bits=16, rate=40000, channels=2, byte_format=1);        

    while(1):
      audioData = audioQueue.get();
      audioQueue.task_done();
      # Play the audio
      pcm.play(audioData);
            
            
def ProcessAudio(audio1,audio2):
    
  # Remove the DC component of the audio
  audio1 = audio1 - np.mean(audio1);
  audio2 = audio2 - np.mean(audio2);
  # Scale the audio
  audio1 = audio1 / np.max([np.abs(audio1),np.abs(audio2)]);
  audio2 = audio2 / np.max([np.abs(audio1),np.abs(audio2)]);
  # Convert to a 16bit signed representation
  audio1 = audio1 * 32768 / 2;
  audio2 = audio2 * 32768 / 2;
  
  return(audio1,audio2);
    
    
def ConvertData(tmp,blkSize):
  # Unpack the bytes of the string in data into unsigned characters
  readFormat = str(blkSize) + 'B'
  tmp = struct.unpack(readFormat,tmp);
  # Convert to a numpy array of floats
  tmp = np.asarray(tmp,dtype=np.float);
  # Subtract 127 from the data (to convert to signed)
  tmp = tmp - 127;
  data = np.zeros(len(tmp)/2, dtype=complex);
  data.real = tmp[::2];
  data.imag = tmp[1::2];
  return data
    
    
def FmDemodulate(data):
  # Calculate the complex vector between two adjacent data points
  tmp = data[1::1] * np.conjugate(data[0:-1:1]);
  # Record the angle of the complex difference vectors
  return np.angle(tmp);
    

def PolyphaseDecimate(filt,inputData,mixValues,filtState,decRate):
    
  # Decompose the input and the filter
  polyFilt = np.reshape(filt,[decRate, -1],order='F');
  polyFilt = np.flipud(polyFilt);
  polyInput = np.reshape(inputData,[decRate,-1],order='F');
  # Pre-allocate the array
  tmp = np.zeros(shape=(decRate,len(inputData)/decRate), dtype=complex);

  # Perform the mixing (only if necessary)
  if len(mixValues) > 0:
    polyMix = np.reshape(mixValues,[decRate,-1],order='F');
    polyInput = polyInput * polyMix;
  
  # Perform the filtering
  for ndx in range(decRate):
    tmp[ndx,:] = lfilter(polyFilt[ndx,:],1,polyInput[ndx,:])
  
  return np.sum(tmp,axis=0);
        
        
def DeEmphasisFilter(audio,fs):
 
  # Calculate the # of samples to hit the -3dB point
  d = fs * 75e-6;
  # Calculate the decay between each sample
  x = np.exp(-1/d);
  # Use this information to create the filter coefficients
  b = [1-x];
  a = [1,-x];
  
  # Perform the filtering and return the result
  return lfilter(b,a,audio);

def PeakFilterDesign(freq,bw):
  # Design a 2nd order IIR peaking filter
  
  # Normalize to pi
  bw = bw*np.pi;
  freq = freq*np.pi;
  
  Gb   = 0.707945784384;
  beta = (Gb/np.sqrt(1-Gb**2))*np.tan(bw/2);
  gain = 1/(1+beta);
  
  num  = (1-gain)*np.array([1,0,-1]);
  den  = np.array([1, -2*gain*np.cos(freq), (2*gain-1)]);
  
  return (num,den);

    
fmDemod = FMDemod();
fileRead = FileReader();
audioPlay = AudioPlay();

dataQueue = Queue.Queue([1]);
audioQueue = Queue.Queue([1]);

sdr = RtlSdr();

# configure device
sdr.sample_rate = 1e6  # Hz
freq = raw_input('Choose a station frequency: ');

try:
  freq = float(freq);
  sdr.center_freq = freq - 200e3;     # Hz
  sdr.gain = 'auto';
  #fileRead.start();
  fmDemod.start();
  audioPlay.start();
  sdr.read_samples_async(sdrCallback, 1024*1000);    
    
except ValueError:
  print("Invalid number");
    

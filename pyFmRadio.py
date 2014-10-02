#!/usr/bin/python

import threading
import Queue
import time
import struct
import ao
import numpy as np
from scipy.signal import lfilter
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
    # Carrier recovery filter
    carrierFilt = [0.000732355054391389,-0.00200004440420536,-0.00187848487221121,-0.00174847123952183,-0.00102068648558740,0.000356908050109597,0.00202694504497847,0.00335394812802437,0.00366374180527268,0.00254187080131728,8.72091515949744e-05,-0.00300699949955061,-0.00561580805574853,-0.00656761649834747,-0.00512796237710826,-0.00139373513873694,0.00358862239841159,0.00807930061878617,0.0102325624609620,0.00882489972562368,0.00384386503427483,-0.00332482162109497,-0.0102451108939210,-0.0142452615109408,-0.0134323991752404,-0.00752553693979967,0.00184370808459819,0.0115451327762795,0.0179920267198243,0.0184664482928355,0.0122351721369207,0.00101121853344407,-0.0114891766142442,-0.0207783796391162,-0.0232242737658299,-0.0174662918034771,-0.00508490190618061,0.00982392129869252,0.0220021989981592,0.0269211927062244,0.0224825081548281,0.00989595991018292,-0.00663084689350798,-0.0213215705565793,-0.0288886782850644,-0.0264695851305936,-0.0147224933310922,0.00234278032171556,0.0187607004930309,0.0287377179350652,0.0287377179350652,0.0187607004930309,0.00234278032171556,-0.0147224933310922,-0.0264695851305936,-0.0288886782850644,-0.0213215705565793,-0.00663084689350798,0.00989595991018292,0.0224825081548281,0.0269211927062244,0.0220021989981592,0.00982392129869252,-0.00508490190618061,-0.0174662918034771,-0.0232242737658299,-0.0207783796391162,-0.0114891766142442,0.00101121853344407,0.0122351721369207,0.0184664482928355,0.0179920267198243,0.0115451327762795,0.00184370808459819,-0.00752553693979967,-0.0134323991752404,-0.0142452615109408,-0.0102451108939210,-0.00332482162109497,0.00384386503427483,0.00882489972562368,0.0102325624609620,0.00807930061878617,0.00358862239841159,-0.00139373513873694,-0.00512796237710826,-0.00656761649834747,-0.00561580805574853,-0.00300699949955061,8.72091515949744e-05,0.00254187080131728,0.00366374180527268,0.00335394812802437,0.00202694504497847,0.000356908050109597,-0.00102068648558740,-0.00174847123952183,-0.00187848487221121,-0.00200004440420536,0.000732355054391389]; 
    # Define the decimation rates at each stage        
    stageOneDec = 5;
    stageTwoDec = 5;         
    
    # Get the next chunk of data
    data = dataQueue.get();
    dataQueue.task_done();
    blkSize = len(data);
    demodBlkSize = len(data) / stageOneDec;
    
    # Downmix the signal, the dongle has a DC filter so we dwell offset
    demodFreq = 200e3;
    # Define the offset between the mono and stereo signals
    stereoDemodFreq = 38e3;
    # Define the raw sampling rate
    fs = 1e6;
    # Create the demodulation signal to baseband the FM station
    sigTime = np.linspace(0,blkSize,blkSize);
    sigTime = sigTime / fs;
    demodSig = np.exp(-1j*2*np.pi*demodFreq*sigTime);
    
    """
    audioTime = np.linspace(0,demodBlkSize,demodBlkSize);
    audioTime = audioTime / fs / stageOneDec;
    stereoDemodSig = np.exp(-1j*2*np.pi*stereoDemodFreq*audioTime);
    """
    
    while(1):
                    
      # Keep only 200KHz of BW
      decData = PolyphaseDecimate(stageOneFilt,data,demodSig,0,stageOneDec);
         
      # Perform the FM Demodulation step
      fmDemod = FmDemodulate(decData);
      # Add an element to keep a proper size for the polyphase filtering
      fmDemod = np.concatenate(([0],fmDemod));
  
      # Filter the input to the audio rate (20KHz)
      audioDataLplusR = np.real(PolyphaseDecimate(stageTwoFilt,fmDemod,[],0,stageTwoDec));
      
      # Scale and DC filter the audio before playing it
      audioDataLplusR = ProcessAudio(audioDataLplusR);

      # Apply the de-emphasis filter
      audioFs = fs/stageOneDec/stageTwoDec;
      audioDataLplusR = DeEmphasisFilter(audioDataLplusR,audioFs);
      
      """            
      audioDataLminusR = np.real(PolyphaseDecimate(stageTwoFilt,fmDemod,stereoDemodSig,0,stageTwoDec));
      
      # Using the two signals (L+R) and (R-L), create the two stereo channels
      audioDataL = audioDataLplusR + audioDataLminusR;
      audioDataR = audioDataLplusR - audioDataLminusR;
      
      # Scale and DC filter the audio before playing it
      audioDataL = ProcessAudio(audioDataL);
      audioDataR = ProcessAudio(audioDataR);

      # Apply the de-emphasis filter
      audioFs = fs/stageOneDec/stageTwoDec;
      audioDataL = DeEmphasisFilter(audioDataL,audioFs);
      audioDataR = DeEmphasisFilter(audioDataR,audioFs);

      # Multiplex the two channels, the ao library assumes this behavior
      # Also convert the data to 16bit signed as the audio stream assumes this format
      stereoAudioData = np.empty(len(audioDataL)*2, dtype=np.int16);
      stereoAudioData[::2] = audioDataL.astype(np.int16);
      stereoAudioData[1::2] = audioDataR.astype(np.int16);
      """
      
      # Put the audio onto the queue and wait for it to be received
      audioQueue.put(audioDataLplusR.astype(np.uint16));
      audioQueue.join();
      
      # Get the next chunk of data
      data = dataQueue.get();
      dataQueue.task_done();
            

class AudioPlay(threading.Thread):
  def run(self):

    # Create the audio device
    pcm = ao.AudioDevice("pulse", bits=16, rate=40000, channels=1, byte_format=1);        
    
    while(1):
      audioData = audioQueue.get();
      audioQueue.task_done();
      # Play the audio
      pcm.play(audioData);
            
            
def ProcessAudio(audio):
    
  # Remove the DC component of the audio
  audio = audio - np.mean(audio);
  # Scale the audio
  audio = audio / np.max(np.abs(audio));
  # Convert to a 16bit signed representation
  return audio * 32768 / 2;
    
    
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
  sdr.center_freq = freq-200e3     # Hz
  sdr.gain = 'auto'
  #fileRead.start();
  fmDemod.start();
  audioPlay.start();
  sdr.read_samples_async(sdrCallback, 1024*1000);    
    
except ValueError:
  print("Invalid number");
    

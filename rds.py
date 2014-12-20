# Copyright 2014 David Swiston
# This program is distributed under the terms of the GNU General Public License v2

#!/usr/bin/python
import numpy as np

class Rds():
  
  def __init__(self,fs):
    # Calculated symbol period 
    self.symbolPeriod = fs/2375.0;
  
  # Raw RDS Input Data
  # ------------------------------------------------------------------------------------------------
  # Data input array
  rbdsData = np.array([],dtype=np.complex64);
  
  # Symbol synchronization variables
  # ------------------------------------------------------------------------------------------------
  # Raw RDS Symbol Data (after symbol synchronization)
  rbdsSymbols = np.array([],dtype=np.complex64);
  # Predicted next sample index (rounded)
  sampValPred = np.array(0,dtype=np.uint32);
  # Predicted next sample index (not rounded)
  sampVal = 0.0;
  
  # Carrier synchronization variables
  # ------------------------------------------------------------------------------------------------
  errFilt = 0.0;
  errVal = 0.0;
  phsVal = np.array(0,dtype=np.float32);
  # Output bitstream
  bits = np.array([],dtype=np.bool);
  
  # Manchester bitstream synchronization and decoding
  # ------------------------------------------------------------------------------------------------
  # Last previous decoded bit, needed to coherently decode across data collections
  decodedBit = None;
  # Decoded outputs, the product of the Manchester decoding process
  decodedBits = np.array([],dtype=np.bool);
  # M of N filter variables used to identify a bitstream synchronization error
  manchMOfN = np.array(np.zeros(8),dtype=np.uint8);
  manchMissNdx = np.array(0,dtype=np.uint8);
  
  # Block specific variables
  # ------------------------------------------------------------------------------------------------
  # Specifies the current location in the input bit buffer
  syncNdx = np.array(0,dtype=np.uint32);
  # Flag to denote of the decoder has block synchronization or not
  sync = np.array(0,dtype=np.bool);
  # Circular M of N buffer used to track successful block decodes
  blockMOfN = np.array(np.zeros(8),dtype=np.uint8);
  # Current index in the M of N buffer
  blockMissNdx = np.array(0,dtype=np.uint8);
  # Specifies the group type of a block set.  It is specified in the second block and needed 
  # when decoding the third and fourth blocks
  groupType = np.array(0,dtype=np.uint32);
  # Similar to group type, specifies the version of a block set.  It is specified in the second 
  # block and needed when decoding the third and fourth blocks
  version = np.array(0,dtype=np.uint32);
  
  # RDS Error Correction Constants
  # ------------------------------------------------------------------------------------------------
  parityCheckMatrix = np.array([512,256,128,64,32,16,8,4,2,1,732,366,183,647,927,787,853,886,443,513,988,494,247,679,911,795],dtype=np.uint16);
  syndromes = np.array([984,980,604,600]);
  generatorPolynomial = np.array(441,dtype=np.uint16);
  
  # RDS Output Data
  # ------------------------------------------------------------------------------------------------
  callSign = str();
  ptyString = str();
  radioText = str();
  
  # RDS Decoding Constants
  rdsPtyLabels = ['None/Undefined',
                  'News',
                  'Information',
                  'Sports',
                  'Talk',
                  'Rock',
                  'Classic Rock',
                  'Adult Hits',
                  'Soft Rock',
                  'Top 40',
                  'Country',
                  'Oldies',
                  'Soft',
                  'Nostalgia',
                  'Jazz',
                  'Classical',
                  'Rhythm and Blues',
                  'Soft Rhythm and Blues',
                  'Language',
                  'Religious Music',
                  'Religious Talk',
                  'Personality',
                  'Public',
                  'College',
                  'Spanish Talk',
                  'Spanish Music',
                  'Hip Hop',
                  'Unassigned',
                  'Unassigned',
                  'Weather',
                  'Emergency Test',
                  'Emergency'];


  def NewData(self,data):
    self.rbdsData = np.append(self.rbdsData,data);


  # Simplistic early-late magnitude only type symbol synchronization instead of Mueller or 
  # Gardner type algorithms.  By using magnitude only, we can perform this operation *before*
  # carrier recovery.
  def SymbolSynchronization(self):

    dataLen = len(self.rbdsData);
    magRbdsData = abs(self.rbdsData);
    sampNdx = np.zeros(dataLen,dtype=np.uint32);
    ndx = 0;    
    
    while(self.sampValPred+1 < dataLen):
      
      # Append the current symbol location estimate
      sampNdx[ndx] = self.sampValPred;
      ndx += 1;
      
      # Using the samples before and after the predicted symbol position, calculate an early
      # or late error determination.
      errVal = -int(magRbdsData[self.sampValPred-1] > magRbdsData[self.sampValPred] ) \
               + int(magRbdsData[self.sampValPred+1] > magRbdsData[self.sampValPred] );
      if (errVal == 0 & (magRbdsData[self.sampValPred-1] > magRbdsData[self.sampValPred])):
        errVal = 1;
        
      # Scale the error to provide some rudimentary filtering.
      errVal = errVal * 0.25;
      
      # Update the predicted next symbol location
      self.sampVal = self.sampVal + errVal + self.symbolPeriod;
      self.sampValPred = round(self.sampVal); 
      
    # Store the symbols using the correct sample points
    self.rbdsSymbols = self.rbdsData[sampNdx[0:ndx]];

    # Clean up and prepare for the next iteration
    # There is no overlap between this block and the next, discard the current block
    if (self.sampValPred-1 >= dataLen):
      self.rbdsData = np.array([],dtype=np.complex64);
      self.sampVal = self.sampVal - dataLen;
      self.sampValPred = self.sampValPred - dataLen;
    # The next symbol estimate overlaps with the current block and the next, save data from this block
    else:
      self.rbdsData = self.rbdsData[self.sampValPred-1:];
      self.sampVal = self.sampVal - self.sampValPred + 1;
      self.sampValPred = 1;
      
    return(sampNdx);


  # The decoding process must be carrier and phase locked before bits can be extracted.  This
  # function performs a simplistic phase lock loop operation.  Note that currently it locks 90
  # degrees out of phase (with a little bit of a bias likely due to timing).  This is why there
  # is a random multiplication by 90 degrees at the end.
  def CarrierSyncronization(self):

    rawBits = np.zeros(len(self.rbdsSymbols),dtype=np.complex64);

    rbdsReal = np.real(self.rbdsSymbols[0]);
    rbdsImag = np.imag(self.rbdsSymbols[0]);
    
    for ndx in range(len(self.rbdsSymbols)):
    
      errSig = rbdsReal * rbdsImag;
      self.errFilt = 0.9 * self.errFilt + 0.1 * errSig;
      
      self.phsVal = self.phsVal + self.errFilt;
      
      rawBits[ndx] = self.rbdsSymbols[ndx] * np.exp(1j*self.phsVal);
      rbdsReal = np.real(rawBits[ndx]);
      rbdsImag = np.imag(rawBits[ndx]);
      
    rawBits = rawBits * np.exp(1j*np.pi/2);
    self.bits = np.append(self.bits,np.real(rawBits) > 0);
    

  # Perform differential Manchester decoding on the bit stream
  def BitDecoding(self):

    # Preallocate the memory required so that we aren't continually growing an array
    numBits = len(self.bits);
    decodedBits = np.zeros(numBits,dtype=np.bool);
    # If this is the first time decoding the rbds buffer, there is no previous bit saved to
    # maintain coherence between buffers.  If it isn't the first time, append the previous bit
    # to the begining of the buffer to maintain coherence.
    if (self.decodedBit == None):
      decodedNdx = 0;
    else:
      decodedNdx = 1;
      decodedBits = np.append(self.decodedBit,decodedBits);

    # Perform the Manchester decoding operation
    bitNdx = 0;
    while bitNdx < len(self.bits)-1:
      # Error detected
      if (self.bits[bitNdx] == self.bits[bitNdx+1]):
        self.manchMOfN[self.manchMissNdx] = 1;
        # Enough errors were detected to signal a synchronization error, slip a bit, reset
        # the M of N filter, and continue
        if (np.sum(self.manchMOfN) > 2):
          bitNdx += 1;
          self.manchMOfN = self.manchMOfN * 0;
      else:
        self.manchMOfN[self.manchMissNdx] = 0;
      decodedBits[decodedNdx] = (self.bits[bitNdx] == 1);
      bitNdx += 2;
      decodedNdx += 1;
      self.manchMissNdx = np.mod(self.manchMissNdx+1,8);
    
    decodedBits = decodedBits[0:decodedNdx];    
    
    # Save off the last bit for future bit decoding
    self.decodedBit = decodedBits[-1];
    
    # Differential decoding
    decodedBits = np.bitwise_xor(decodedBits[1:],decodedBits[0:-1]);
    self.decodedBits = np.append(self.decodedBits,decodedBits);
    
    #
    self.bits = self.bits[bitNdx:];
    

  def ProcessBlocks(self):

    # This function assumes new data has been placed into the buffer queue
    if (len(self.rbdsData) == 0):
      return;
      
    # Perform symbol synchronization first
    self.SymbolSynchronization();
    # Synchronize to the carrier
    self.CarrierSyncronization();
    # Perform bit decoding via Manchester decoding
    self.BitDecoding();
    
    # Loop over all the data in the queue and extract the block contents
    while (self.syncNdx < len(self.decodedBits)-26):
      # First things first, if we are not block locked or lost block lock for some reason, obtain it
      if (self.sync == 0):
        self.SyncToBlock();
        continue;
        
      # Extract data from the current block
      self.DecodeBlock();
      
    # Save unused bits from the last partial block for the next decode
    self.decodedBits = self.decodedBits[self.syncNdx:];
    # Reset the sync index for the next time around
    self.syncNdx = 0;


  # Block search.  Move bit by bit and try to identify a valid block syndrome
  def SyncToBlock(self):

    # Loop over all the decoded bits looking for a valid syndrome
    while (self.syncNdx < len(self.decodedBits)-26):
      syndrome = self.CalculateSyndrome( self.decodedBits[self.syncNdx:self.syncNdx+26] );
      # Once a valid syndrome is found, indicate synchronization, and record the block located
      if (np.any(syndrome == self.syndromes)):
        self.blockNdx = np.flatnonzero(syndrome == self.syndromes);
        self.sync = 1;
        break;
      self.syncNdx = self.syncNdx + 1;


  def DecodeBlock(self):

    syndrome = self.CalculateSyndrome( self.decodedBits[self.syncNdx:self.syncNdx+26] );  

    # First, check to see if there was a bit error or there is a block synchronization error
    if (syndrome == self.syndromes[self.blockNdx]):
      # There is no error
      # Indicate no error in the MofN counter
      self.blockMOfN[self.blockMissNdx] = 0;
      # Decode the block
      if (self.blockNdx == 0):
        self.DecodeBlockZero();
      elif (self.blockNdx == 1):
        self.DecodeBlockOne();
      elif ((self.blockNdx == 2) | (self.blockNdx == 3)):
        self.DecodeBlockTwoAndThree();
      # Increment the sync and block indices in preparation for the next block.
      self.syncNdx += 26;
      self.blockNdx += 1;
      self.blockMissNdx += 1;

    # The syndrome did not match expected.  Either encountered a bit error or we have a block 
    # synchronization error.
    else:
      
      # Try to correct the error
      syndrome = np.bitwise_xor(syndrome,self.syndromes[self.blockNdx]);
      (self.decodedBits[self.syncNdx:self.syncNdx+26],corrected) = self.ErrorCorrection(syndrome, self.decodedBits[self.syncNdx:self.syncNdx+26]);
      
      # If we were able to correct the error, return without incrementing the buffer location so we 
      # reprocess the block
      if (corrected == 1):
        return;
      else:
        # May have been an unrecoverable bit error OR the checkbits (last 10 bits) contained an error
        self.blockMOfN[self.blockMissNdx] = 1;
        # The M of N filter has signaled an error, go back to the block search mode
        if (np.sum(self.blockMOfN) > 2):
          self.sync = 0;
          self.blockMOfN = self.blockMOfN * 0;
          self.ResetBlockInfo();
        # There was an uncorrectable error but not enough errors have been caught to reset
        # to block search mode.  Increment the sync and block indices in preparation for
        # the next block.
        else:
          self.syncNdx += 26;
          self.blockNdx += 1;
          self.blockMissNdx += 1;
    
    # Update pointers to the next M of N buffer location and the next block location
    self.blockMissNdx = np.mod(self.blockMissNdx,8);
    self.blockNdx = np.mod(self.blockNdx,4);


  # Using the parity check matrix, calculate the syndrome for a block
  def CalculateSyndrome(self,bits):

    syndrome = 0;
    for ndx in range(26):
      if bits[ndx]:
        syndrome = np.bitwise_xor(syndrome, self.parityCheckMatrix[ndx]);
    return syndrome;


  def DecodeBlockZero(self):
    pi = self.ArrayBinaryToDecimal(self.decodedBits[self.syncNdx:self.syncNdx+16]);
    self.CalculateStation(pi);
    
    
  def DecodeBlockOne(self):
    self.groupType = self.ArrayBinaryToDecimal(self.decodedBits[self.syncNdx:self.syncNdx+4]);
    self.version = self.decodedBits[self.syncNdx+4];
    pty = self.ArrayBinaryToDecimal(self.decodedBits[self.syncNdx+6:self.syncNdx+11]);
    self.ptyString = self.rdsPtyLabels[pty];
    
    if ((self.groupType == 2) & (self.version == 0)):
      self.radioTxtLoc = self.ArrayBinaryToDecimal(self.decodedBits[self.syncNdx+12:self.syncNdx+16]);
      
      
  def DecodeBlockTwoAndThree(self):
    if ((self.groupType == 2) & (self.version == 0)):
      character1 = chr(self.ArrayBinaryToDecimal(self.decodedBits[self.syncNdx:self.syncNdx+8]));
      character2 = chr(self.ArrayBinaryToDecimal(self.decodedBits[self.syncNdx+8:self.syncNdx+16]));
      messageNdx = self.radioTxtLoc * 4 + (self.blockNdx-2)*2;
      self.radioText = self.radioText[0:messageNdx] + character1 + character2 + self.radioText[messageNdx+2:];
 

  # The RBDS specification has a shortened cyclic (26,16) block code that is able to correct
  # a random burst error of up to 5 bits.  It has additional detection capabilities.
  def ErrorCorrection(self,syndrome,inBits):
  # LUT based error correction
  # errVector = find( errorSyndromes == syndrome );
  # if numel(errVector)>1
  #   disp('Help!');
  # end
  # if isempty(errVector)
  #   failure = true;
  # else
  #   inBits = bitxor(inBits,errorVectors(errVector(1),:));
  #   failure = false;
  # end

    corrected = 0;
    if(syndrome):
      # Meggitt algorithm, only correct data bits (16), not the parity bits
      for ndx in range(16):
        
        # The first (most significant) bit is one
        if (np.bitwise_and(syndrome,512)):
          # If the first bit is a one and the last 5 (least significant bits)
          # are zero, this indicates an error at the current bit (ndx) position
          if (np.bitwise_and(syndrome,31) == 0):
            # The code can correct bursts up to 5 bits long.  Check to see if
            # the error is a burst or not.  If it isn't a burst, it isn't
            # correctable, return immediately with a failure.
            tmp = np.bitwise_and(syndrome,480);
            if ~(tmp == 480 | tmp == 448 | tmp == 384 | tmp == 256 | tmp == 0):
              break;
            # The error appears to be a burst error, attempt to correct
            inBits[ndx] = np.bitwise_xor(inBits(ndx),1);
            # Shift the syndrome
            syndrome = np.left_shift(syndrome,1);
          else:
            # Least significant bits do not indicate the current bit (ndx) is
            # an error in a burst.  Continue shifting the syndrome and then apply the
            # generator polynomial.
            syndrome = np.left_shift(syndrome,1);
            syndrome = np.bitwise_xor(syndrome,441);
        else:
          # Not a one at the first (most significant) syndrome bit, applying generator polynomial
          # is trivial in this case.
          syndrome = np.left_shift(syndrome,1);
      # If after this process the syndrome is not zero, there was a
      # uncorrectable error.
      if (np.bitwise_and(syndrome,1023)==0):
        corrected = 1;
    return (inBits, corrected);


  # Take the decimal pi value and convert it to the station's call sign
  def CalculateStation(self,pi):

    usKoffset = 4096;
    usWoffset = 21672; # usKoffset + 26*26*26;
    usWtop = usWoffset + 26*26*26;

    if(pi > usWtop):
      self.callSign = 'ERROR';
      return;
    elif (pi < usWoffset):
      self.callSign = 'K';
      pi = pi - usKoffset;
    else:
      self.callSign = 'W';
      pi = pi - usWoffset;
  
    val = str();
    for ndx in range(3):
      val = str(chr(ord('A') + np.mod(pi,26))) + val;
      pi = pi / 26;
  
    self.callSign = self.callSign + val;


  def ResetBlockInfo(self):
    self.groupType = 0;
    self.version = 0;


  # Helper function to convert an array of binary values to a decimal value.  Accepts all sorts of
  # flags/formats including power offsets, different endianness, and twos complement formats
  def ArrayBinaryToDecimal(self, inputArray, powOffset = 0, endianness = 0, twosComplement = 0):
    
    if (np.ndim(inputArray) == 1):
      inputArray = np.expand_dims(inputArray,0);
      
    inputLen = np.size(inputArray,1);
      
    if (twosComplement):
      inputLen = inputLen - 1;
    
    if (endianness == 0):
      if (twosComplement):
        offset = -(2**(inputLen+powOffset)) * inputArray[:,0];
        inputArray = inputArray[:,1:];
      else:
        offset = 0;
  
      powArray = 2**np.arange(powOffset+inputLen-1,powOffset-1,-1,dtype=np.float64);
  
    else:
      if (twosComplement):
        offset = -(2**inputLen) * inputArray[:,-1];
        inputArray = inputArray[:,1:-1];
      else:
        offset = 0;
  
      powArray = 2**np.arange(powOffset,inputLen,dtype=np.float64);
  
    outDec = np.transpose(np.dot(powArray,inputArray.T)) + offset;
  
    if twosComplement:
      return np.int32(outDec);
    else:
      return np.uint32(outDec);
    

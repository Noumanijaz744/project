import random
import numpy as np
import librosa

class SpecAugment():
    
    def __init__(self, signal, sr, policy, zero_mean_normalized=True):
        self.signal = signal
        self.sr = sr
        self.policy = policy
        self.zero_mean_normalized = zero_mean_normalized
        
        # Policy Specific Parameters
        if self.policy == 'LB':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 1, 100, 1.0, 1
        elif self.policy == 'LD':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 80, 27, 2, 100, 1.0, 2
        elif self.policy == 'SM':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 15, 2, 70, 0.2, 2
        elif self.policy == 'SS':
            self.W, self.F, self.m_F, self.T, self.p, self.m_T = 40, 27, 2, 70, 0.2, 2
        
        # Compute MFCC
        self.mfcc = librosa.feature.mfcc(self.signal, sr=self.sr, n_mfcc=40)
        if self.zero_mean_normalized:
            self.mfcc = librosa.util.normalize(self.mfcc, norm=2)
    
    def freq_mask(self):
        
        v = self.mfcc.shape[0] # no. of mel bins
        
        # apply m_F frequency masks to the MFCC
        for i in range(self.m_F):
            f = int(np.random.uniform(0, self.F)) # [0, F)
            f0 = random.randint(0, v - f) # [0, v - f)
            self.mfcc[f0:f0 + f, :] = 0
            
        return self.mfcc
    
    
    def time_mask(self):
    
        tau = self.mfcc.shape[1] # time frames
        
        # apply m_T time masks to the MFCC
        for i in range(self.m_T):
            t = int(np.random.uniform(0, self.T)) # [0, T)
            t0 = random.randint(0, tau - t) # [0, tau - t)
            self.mfcc[:, t0:t0 + t] = 0
            
        return self.mfcc
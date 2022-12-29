import pywt

from biosppy import ecg, tools
import numpy as np
import pandas as pd
import scipy

class Extracter():
    all=[]
    lead_features=0

    def cal_entropy(self,coeff):
        coeff = pd.Series(coeff).value_counts()
        entropy = scipy.stats.entropy(coeff)
        return entropy / 10
    def cal_statistics(self,signal):
        n5 = np.percentile(signal, 5)
        n25 = np.percentile(signal, 25)
        n75 = np.percentile(signal, 75)
        n95 = np.percentile(signal, 95)
        median = np.percentile(signal, 50)
        mean = np.mean(signal)
        std = np.std(signal)
        var = np.var(signal)
        return [n5, n25, n75, n95, median, mean, std, var]
    def extract_lead(self,signal, sampling_rate):
        pass

    def extract(self,ecg_data, sampling_rate=500):
        self.all=[]
        for signal in ecg_data.T:
            all+=self.extract_lead(signal,sampling_rate)
        return all

class Extract_feature(Extracter):
    def cal_entropy(self, coeff):
        pass
    def cal_statistics(self, signal):
        pass
    def extract(self, ecg_data, sampling_rate=500):
        pass
    def extract_lead(self, signal, sampling_rate):
       # extract expert features for single-lead ECGs: statistics, shannon entropy
        self.lead_features = self.cal_statistics(signal) # statistic of signal
        coeffs = pywt.wavedec(signal, 'db10', level=4)
        for coeff in coeffs:
            lead_features.append(self.cal_entropy(coeff)) # shannon entropy of coefficients
            lead_features += self.cal_statistics(coeff) # statistics of coefficients
        return lead_features

class Extract_heart_rates(Extracter):
    def cal_entropy(self, coeff):
        pass
    def cal_statistics(self, signal):
        pass
    def extract(self, ecg_data, sampling_rate=500):
        pass
    def extract_lead(self, signal, sampling_rate):
        # extract heart rate for single-lead ECG: may return empty list
        rpeaks, = ecg.hamilton_segmenter(signal=signal, sampling_rate=sampling_rate)
        rpeaks, = ecg.correct_rpeaks(signal=signal, rpeaks=rpeaks, sampling_rate=sampling_rate, tol=0.05)
        _, heartrates = tools.get_heart_rate(beats=rpeaks, sampling_rate=500, smooth=True, size=3)
        return list(heartrates / 100) # divided by 100    

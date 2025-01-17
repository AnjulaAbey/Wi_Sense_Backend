from django.utils.timezone import now, timedelta
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
import numpy as np
import pandas as pd
from scipy.signal import ellip, filtfilt, welch
from sklearn.decomposition import PCA
from cms.models import CSIData
from cms.serializers import CSIDataSerializer
from datetime import datetime
from datetime import datetime, timedelta
from django.utils.timezone import make_aware


ESP32_start_time = datetime(2025, 1, 9, 9, 5) 
class CSIDataViewSet(viewsets.ModelViewSet):
    queryset = CSIData.objects.all()
    serializer_class = CSIDataSerializer

    def smooth(a, WSZ):
        out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid') / WSZ
        r = np.arange(1, WSZ - 1, 2)
        start = np.cumsum(a[:WSZ - 1])[::2] / r
        stop = (np.cumsum(a[:-WSZ:-1])[::2] / r)[::-1]
        return np.concatenate((start, out0, stop))

    def movingAverage(signal, window):
        C2 = np.zeros((signal.shape[0], signal.shape[1]))
        for i in range(signal.shape[1]):
            C2[:, i] = CSIDataViewSet.smooth(signal[:, i], window)
        return C2

    def load_and_preprocess(data):
        # Process raw CSI data
        index = list(range(len(data)))
        AmpCSI = np.zeros((len(data), 64))
        PhaseCSI = np.zeros((len(data), 64))
        # print(index)
        a=0
        for i in index:
            # parts = data.spilit(',')
            # rawCSI=[s.strip('[') for s in parts]
            # rawCSI=[s.strip('[') for s in parts]
            # rawCSI=[s.strip(']') for s in rawCSI]
            #rawCSI.pop()
            ImCSI=np.array(data[a][::2],dtype=np.int64)
            ReCSI=np.array(data[a][1::2],dtype=np.int64)
            AmpCSI[i][:]=np.sqrt(np.power(ImCSI[:],2) + np.power(ReCSI[:],2))
            PhaseCSI[i][:]= np.arctan2(ImCSI[:], ReCSI[:])
            a=a+1
        Amp = np.concatenate((AmpCSI[:, 6:32], AmpCSI[:, 33:59]), axis=1)
        signal_dc_removed = Amp - np.mean(Amp, axis=0)
        smoothed_signal = CSIDataViewSet.movingAverage(signal_dc_removed, window=11)
        return smoothed_signal

    def ellip_bandpass(signal, lowcut, highcut, fs, order=4, rp=1, rs=40):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = ellip(order, rp, rs, [low, high], btype='band')
        filtered_signal = np.zeros_like(signal)
        for i in range(signal.shape[1]):
            filtered_signal[:, i] = filtfilt(b, a, signal[:, i])
        return filtered_signal

    def estimate_respiration_rate(signal, fs):
        pca = PCA()
        pca_components = pca.fit_transform(signal)
        first_pc = pca_components[:, 0]
        freqs, psd = welch(first_pc, fs=fs, nperseg=len(first_pc))
        peak_freq = freqs[np.argmax(psd)]
        rr_bpm = peak_freq * 60
        return rr_bpm

    @action(detail=False, methods=["get"])
    def predict_respiration_rate(self, request):
        # Get the current time in UTC
        now = datetime(2025, 1, 9, 9, 10) 
        
        # Calculate the time window you want (from now to one minute ago)
        one_minute_ago = now - timedelta(minutes=2)
        # Calculate the relative time (in microseconds) for now and one minute ago
        now_duration = int((now - ESP32_start_time).total_seconds() * 1e6)  # Convert to microseconds
        one_minute_ago_duration = int((one_minute_ago - ESP32_start_time).total_seconds() * 1e6)  # Convert to microseconds
        print("now " , now_duration)
        print("one minute ago " , one_minute_ago_duration)
        # Fetch the CSI data between the two durations (from one minute ago to now)
        latest_data = CSIData.objects.filter(
            time_stamp__gte=str(one_minute_ago_duration),  # Time duration one minute ago
            time_stamp__lte=str(now_duration)  # Time duration now
        ).values_list("raw_data", flat=True)
        
        print(len(latest_data))
        time_range= (now_duration - one_minute_ago_duration)
        print(time_range/10**6)
        if not latest_data.exists():
            return Response({"error": "No data available for the last minute."}, status=400)
        
        # Preprocess the data
        csi_data = list(latest_data)
        preprocessed_signal = CSIDataViewSet.load_and_preprocess(csi_data)
        
        # Calculate sampling frequency
        fs = preprocessed_signal.shape[0] / 60
        print(fs)
        
        # Apply bandpass filter
        filtered_signal = CSIDataViewSet.ellip_bandpass(preprocessed_signal, lowcut=0.18, highcut=0.65, fs=fs)
        print(filtered_signal)
        
        # Estimate respiration rate
        rr_bpm = CSIDataViewSet.estimate_respiration_rate(filtered_signal, fs=fs)

        return Response({"respiration_rate_bpm": rr_bpm})
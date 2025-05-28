from django.utils.timezone import now, timedelta
from rest_framework import viewsets, status
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
import joblib
from .utils import hampel_filter_fast, Get_Amp, apply_pca

model = joblib.load('presence_detection_model.pkl')

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
            # Fetch the latest 6000 CSI entries, ordered by timestamp (most recent first)
        latest_data = CSIData.objects.order_by('-port_time_stamp')[:6000].values_list("raw_data", flat=True)
        latest_entries = CSIData.objects.order_by('-port_time_stamp')[:6000].values("time_stamp")
        if not latest_data:
            return Response({"error": "No CSI data available."}, status=status.HTTP_400_BAD_REQUEST)
            # Extract timestamps for fs calculation
        latest_entries = list(latest_entries)
        start_time = int(latest_entries[0]["time_stamp"])
        end_time = int(latest_entries[-1]["time_stamp"])
        time_duration_microseconds = end_time - start_time
        time_duration_seconds = time_duration_microseconds / 1e6
        print(time_duration_seconds)
        # Reverse to chronological order
        csi_data = list(latest_data)[::-1]
        preprocessed_signal = CSIDataViewSet.load_and_preprocess(csi_data)
        
        # Calculate sampling frequency
        fs = preprocessed_signal.shape[0] / time_duration_seconds
        print(fs)
        
        # Apply bandpass filter
        filtered_signal = CSIDataViewSet.ellip_bandpass(preprocessed_signal, lowcut=0.18, highcut=0.65, fs=fs)
        # print(filtered_signal)
        
        # Estimate respiration rate
        rr_bpm = CSIDataViewSet.estimate_respiration_rate(filtered_signal, fs=fs)

        return Response({"respiration_rate_bpm": rr_bpm})
    
class RealTimePresenceDetection(viewsets.ModelViewSet):
    queryset = CSIData.objects.all().order_by('-port_time_stamp')
    serializer_class = CSIDataSerializer

    @action(detail=False, methods=['get'], url_path='predict')
    def predict_presence(self, request):
        # try:
            # Fetch latest 30 entries
            latest_entries = CSIData.objects.order_by('-port_time_stamp')[:100].values()
            # print(latest_entries)
            latest_entries = pd.DataFrame(latest_entries) 
            print(latest_entries)
            #  # Ensure time order
            # # print(latest_entries.head())
            

            # # Step 3: Create DataFrame from combined data
            df = latest_entries
            # # Apply preprocessing
            df["port_time_stamp"] =  df["port_time_stamp"].astype(int)
            df["time_stamp"] =  df["time_stamp"].astype(int)
            df["CSI_DATA"] = df["raw_data"]
            signal, fs, time, _, _ = Get_Amp(df)
            filtered = hampel_filter_fast(pd.DataFrame(signal))
            # # latest = filtered.iloc[-1:].values  # Last sample for prediction
            pca_features = apply_pca(filtered, 20, explained_variance=0.95)
            # # print(pca_features[0])
            # # # Predict
            # # pca_features.shape()
            prediction = model.predict(pca_features[0])
            print(prediction)
            # # presence = int(prediction[0])
            # # print(presence)
            return Response({"presence": prediction}, status=status.HTTP_200_OK)
        
        # except Exception as e:
        #     return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

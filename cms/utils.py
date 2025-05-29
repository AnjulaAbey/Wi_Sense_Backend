import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import skew, kurtosis, mode

def Get_Amp(data):
    if len(data) <= 1:
        return None

    # Time range
    time_range = data["time_stamp"].iloc[0] - data["time_stamp"].iloc[-1]

    # Prepare
    # csi_raw = " ".join(data["CSI_DATA"])
    num_rows = len(data["CSI_DATA"])
    AmpCSI = np.zeros((num_rows, 64))
    PhaseCSI = np.zeros((num_rows, 64))

    # Process CSI rows
    for i in range(num_rows - 1):  # last row skipped as in original
        # Clean and split
        # parts = csi_raw[i].split()
        # parts = np.array(data["CSI_DATA"], dtype=np.int64)
        parts = np.array(data["CSI_DATA"].iloc[i])
        print(parts)
        
        # Separate real and imaginary
        ImCSI = parts[::2]
        ReCSI = parts[1::2]

        # Compute amplitude and phase
        AmpCSI[i, :] = np.sqrt(ImCSI**2 + ReCSI**2)
        PhaseCSI[i, :] = np.arctan2(ImCSI, ReCSI)

    # Remove subcarriers 32 (index 32) from both Amp and Phase
    Amp = np.concatenate((AmpCSI[:, 6:32], AmpCSI[:, 33:59]), axis=1)
    Pha = np.concatenate((PhaseCSI[:, 6:32], PhaseCSI[:, 33:59]), axis=1)

    # DC removal
    signal_dc_removed = Amp - np.mean(Amp, axis=0)

    # Prepare dataset
    # y = pd.Series(data["Presence"].values[:len(signal_dc_removed)], name='y')
    x = pd.DataFrame(signal_dc_removed)
    y = None
    # x_y_dataset = pd.concat([x, y], axis=1)
    x_y_dataset = x

    # Sampling frequency
    fs = signal_dc_removed.shape[0] / time_range
    time = np.arange(signal_dc_removed.shape[0]) / fs

    # Optional plot
    # plt.plot(time, signal_dc_removed)
    
    return signal_dc_removed, fs, time, x_y_dataset, y

def hampel_filter_fast(df, window_size=5, n_sigmas=3):
    """
    Applies the Hampel filter to each column of the dataframe using vectorized operations.
    Parameters:
        df: pd.DataFrame - Each column is a subcarrier signal.
        window_size: int - Half window size for rolling.
        n_sigmas: int - Threshold multiplier for outlier detection.
    Returns:
        filtered_df: pd.DataFrame - Filtered data.
    """
    k = 1.4826  # Scale for Gaussian distribution

    def hampel_single_column(col):
        rolling_median = col.rolling(window=2*window_size+1, center=True).median()
        mad = col.rolling(window=2*window_size+1, center=True).apply(
            lambda x: median_abs_deviation(x, scale=k), raw=True
        )
        difference = np.abs(col - rolling_median)
        outlier_mask = difference > (n_sigmas * mad)
        filtered = col.copy()
        filtered[outlier_mask] = rolling_median[outlier_mask]
        return filtered

    # Apply to each column
    return df.apply(hampel_single_column, axis=0)

def apply_pca(data, n_components=None, explained_variance=0.95):
    # Separate features from timestamps (if present)
    if 'start_timestamp' in data.columns:
        timestamps = data['start_timestamp']
        features = data.drop(columns=['start_timestamp'])
    else:
        features = data

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Initialize PCA
    if n_components is None:
        pca = PCA(n_components=explained_variance)  # Retain explained variance
    else:
        pca = PCA(n_components=n_components)  # Retain specific number of components

    # Fit and transform the data
    principal_components = pca.fit_transform(scaled_features)

    # Create a DataFrame for the principal components
    pca_df = pd.DataFrame(
        principal_components,
        columns=[f"PC{i+1}" for i in range(principal_components.shape[1])]
    )

    # Add timestamps back if present
    if 'start_timestamp' in data.columns:
        pca_df['start_timestamp'] = timestamps.values

    return pca_df, pca

def extract_features_with_sliding_window(data, sampling_rate, window_length, y):
    # Determine the window size in samples
    window_size = int(sampling_rate * window_length)
    print(window_size)
    print(data)
    step_size = 1  # No overlap; adjust if overlap is desired

    features = []
    subcarrier_cols = data  # Subcarrier groups

    # for start in range(0, len(data) - window_size + 1, step_size):
    #     window = data[start : start + window_size]  # Extract sliding window

    #     window_features = {}
    #     for i in range(len(subcarrier_cols[0])):
    #         col = i
    #         subcarrier_data = window[i]
    #         window_features[f"{col}_mean"] = subcarrier_data.mean()
    #         window_features[f"{col}_max"] = subcarrier_data.max()
    #         window_features[f"{col}_min"] = subcarrier_data.min()
    #         window_features[f"{col}_var"] = subcarrier_data.var()
    #         window_features[f"{col}_skew"] = skew(subcarrier_data)
    #         window_features[f"{col}_range"] = subcarrier_data.max() - subcarrier_data.min()
    #         window_features[f"{col}_mode"] = mode(subcarrier_data, nan_policy='omit')[0]
    #         window_features[f"{col}_median"] = np.median(subcarrier_data)
    #         window_features[f"{col}_kurtosis"] = kurtosis(subcarrier_data)
    #     window_features["y"] = y[start + window_size-1]

    #     features.append(window_features)

    # Convert to a DataFrame
    features_df = pd.DataFrame(features)
    return features_df 


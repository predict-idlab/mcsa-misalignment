import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import scipy.signal as ss
from scipy.stats import skew, kurtosis
import antropy as ant
import matplotlib.mlab as mlab
from tsflex.features.integrations import tsfresh_settings_wrapper
from tsflex.features import MultipleFeatureDescriptors, FeatureCollection, FuncWrapper


def zip_dicts(*dcts):
    """
    Zips dictionaries
    """
    if not dcts:
        return
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)


def read_files(folder_name):
    """
    Reads all files in a folder and returns a dataframe with the data.

    Args:
        folder_name (str): path to the folder

    Returns:
        pd.DataFrame: dataframe with the data
    """
    # Create a list of files in the folder
    files = os.listdir(folder_name)
    # Loop through the files and read them into the dataframe
    return pd.concat([pd.read_csv(folder_name + file) for file in tqdm(files)])


def segment_df(raw_df, window_size=1000):
    """
    This function segments a dataframe into windows of the defined amount of samples and adds a window id as a column.

    Args:
        raw_df (pd.DataFrame): dataframe to segment
        window_size (int, optional): window size in samples. Defaults to 1000.

    Returns:
        pd.DataFrame: segmented dataframe
    """
    
    df_list = []
    for name, group in tqdm(raw_df.groupby(['misalignment', 'recording_nr', 'direction', 'speed'])):
        # Segment group into windws and add window id as a column
        group['window_id'] = np.arange(len(group)) // window_size
        
        # Check if last window is smaller than the specified window size
        if len(group) % window_size != 0:
            # Omit last window
            group = group[group['window_id'] < group['window_id'].max()]
        # Add the segmented group to the list
        df_list.append(group)
        
    return pd.concat(df_list)


def find_peaks(sig):
    """
    Finds peaks in a signal.

    Args:
        sig (np.array): signal

    Returns:
        np.array: indices of the peaks
    """
    peaks_high, _ = ss.find_peaks(sig, height=0.001, distance=50, width=50)
    peaks_low, _ = ss.find_peaks(sig*-1, height=0.001, distance=50, width=50)
    return peaks_high, peaks_low


def get_y(params, x):
    """
    Calculates the y value for a given x value and a set of parameters.

    Args:
        params (np.array): parameters
        x (np.array): x values

    Returns:
        np.array: y values
    """
    deg = len(params) - 1
    xs = np.array([x**d for d in range(deg,0,-1)])
    xs *= np.expand_dims(params[:-1], axis=1)
    return np.sum(xs, axis=0) + params[-1]


def fit_peak(s, peak_idx, window_size=15, deg=2):
    """
    Fits a polynomial to a peak.

    Args:
        s (np.array): signal
        peak_idx (int): index of the peak
        window_size (int, optional): window size. Defaults to 15.
        deg (int, optional): degree of the polynomial. Defaults to 2.

    Returns:
        np.array: parameters of the polynomial
    """
    if peak_idx < window_size or peak_idx + window_size + 1 >= len(s):
        return None
    s = s[peak_idx-window_size:peak_idx+window_size+1]
    return np.polyfit(np.arange(-window_size,window_size+1), s, deg=deg)


def extract_manual_features(df, window_size, signal):
    """
    Extracts manual features from a dataframe.

    Args:
        df (pd.DataFrame): dataframe
        window_size (int): window size
        signal (str): signal to extract features from

    Returns:
        pd.DataFrame: dataframe with the extracted features
    """
    perm_list = []
    feat_list = []

    for i, (name, group) in tqdm(enumerate(df.groupby(['misalignment', 'recording_nr', 'direction', 'speed', 'window_id']))):
        data = group.reset_index()
        signal_df = data[signal]

        # TODO: what if no peaks are found
        peaks_h_idx, peaks_l_idx = find_peaks(signal_df)
        start = peaks_h_idx[0]
        end = peaks_h_idx[-1]
        cut_signal_df = signal_df[start:end]
        peaks_h = signal_df[peaks_h_idx]
        peaks_l = signal_df[peaks_l_idx]
        len_peaks = min([len(peaks_h), len(peaks_l)])
        feats_h = [len(peaks_h), np.std(peaks_h), np.mean(peaks_h), *np.quantile(peaks_h, q=[0, 0.25, 0.5, 0.75, 1]), skew(peaks_h), kurtosis(peaks_h), max(peaks_h) - min(peaks_h)]#, iqr(peaks_h)]
        feats_l = [len(peaks_l), np.std(peaks_l), np.mean(peaks_l), *np.quantile(peaks_l, q=[0, 0.25, 0.5, 0.75, 1]), skew(peaks_l), kurtosis(peaks_l), max(peaks_l) - min(peaks_l)]#, iqr(peaks_h)]
        peaks_diff = np.array(peaks_h[:len_peaks]) - np.array(peaks_l[:len_peaks])
        feats_diff = [len(peaks_diff), np.std(peaks_diff), np.mean(peaks_diff), *np.quantile(peaks_diff, q=[0, 0.25, 0.5, 0.75, 1]), skew(peaks_diff), kurtosis(peaks_diff), max(peaks_diff) - min(peaks_diff)]#, iqr(peaks_h)]

        peak_params_stats = []

        peak_params = []; w = 20;
        for p in peaks_h_idx:
            params = fit_peak(signal_df, p, window_size=w, deg=4)
            peak_params += [list(params)]
        peak_params_stats += [*np.std(peak_params, axis=0), *np.max(peak_params, axis=0), *np.min(peak_params, axis=0)]
        peak_params = []; w = 20;
        for p in peaks_l_idx:
            params = fit_peak(signal_df, p, window_size=w, deg=4)
            peak_params += [list(params)]
        peak_params_stats += [*np.std(peak_params, axis=0), *np.max(peak_params, axis=0), *np.min(peak_params, axis=0)]

        spec, freqs = mlab.magnitude_spectrum(x=cut_signal_df)
        spec_ph, freqs_ph = mlab.phase_spectrum(x=cut_signal_df)
        spec_ang, freqs_ang = mlab.angle_spectrum(x=cut_signal_df)
        assert all(np.isclose(freqs, freqs_ph))
        assert all(np.isclose(freqs, freqs_ang))
        signal_specs = list(spec[np.argsort(spec)[::-1]][:5]) + list(spec_ph[np.argsort(spec)[::-1]][:5]) + list(spec_ang[np.argsort(spec)[::-1]][:5])
        NFFT = 2**int(np.ceil(np.log2(window_size)))

        spec_signal, _ = mlab.psd(signal_df, NFFT=NFFT)
        signal_specs += [np.max(spec_signal), np.min(spec_signal)]    

        feat_list.append(feats_h + feats_l + feats_diff + peak_params_stats  + signal_specs)
        perm_list += [name]

    feat_names = ["len", "std", "mean"] + ["min", "q_.25", "med", "q_.75", "max"] + ["skew", "kurt", "gap"]
    feat_names = [f+f"_{s}" for s in ["peak_h", "peak_l", "peak_diff"] for f in feat_names]
    feat_names += [f"peak_params_{i}" for i in range(len(peak_params_stats))]
    feat_names += [f"spec_{i}" for i in range(len(signal_specs))]

    # Add signal prefix to feature names
    feat_names = [signal + "__" + f for f in feat_names]
    feat_df = pd.DataFrame(feat_list, columns=feat_names)
    return perm_list, feat_df


def extract_residual_features(df, window_size, signal_names=["high_res", "low_res"]):
    """
    Extracts residual features from a dataframe.

    Args:
        df (pd.DataFrame): dataframe
        window_size (int): window size
        signal_names (list, optional): names of the signals. Defaults to ["high_res", "low_res"].

    Returns:
        pd.DataFrame: dataframe with the extracted features
    """
    
    perm_list = []
    feat_list = []
    
    for name, group in tqdm(df.groupby(['misalignment', 'recording_nr', 'direction', 'speed', 'window_id'])):
        data = group.reset_index()

        high_res_df = data[signal_names[0]]
        low_res_df = data[signal_names[1]]

        peaks_h_idx, peaks_l_idx = find_peaks(high_res_df)
        start = peaks_h_idx[0]
        end = peaks_h_idx[-1]

        params = np.polyfit(high_res_df, low_res_df, deg=1)
        residual = (low_res_df - get_y(params, high_res_df))[start:end]
        residual_stats = [len(residual), np.std(residual), np.mean(residual), *np.quantile(residual, q=[0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95 ,0.98 ,0.99, 1]), skew(residual), kurtosis(residual)] #+ diff_stats(residual)

        spec, freqs = mlab.magnitude_spectrum(x=residual)
        spec_ph, freqs_ph = mlab.phase_spectrum(x=residual)
        spec_ang, freqs_ang = mlab.angle_spectrum(x=residual)
        assert all(np.isclose(freqs, freqs_ph))
        assert all(np.isclose(freqs, freqs_ang))
        residual_specs = list(spec[np.argsort(spec)[::-1]][:5]) + list(spec_ph[np.argsort(spec)[::-1]][:5]) + list(spec_ang[np.argsort(spec)[::-1]][:5])

        NFFT = 2**int(np.ceil(np.log2(window_size)))
        spec_residual, _ = mlab.psd(residual, NFFT=NFFT)
        residual_specs += [np.max(spec_residual), np.min(spec_residual)]    

        feat_list.append(list(params) + residual_stats + residual_specs)
        perm_list += [name]

    feat_names = ["slope", "intercept"]
    feat_names += [f"stat_{i}" for i in range(len(residual_stats))]
    feat_names += [f"spec_{i}" for i in range(len(residual_specs))]

    # Add residual prefix to feature names
    feat_names = ["residual__" + f for f in feat_names]
    feat_df = pd.DataFrame(feat_list, columns=feat_names)
    return perm_list, feat_df


def check_peaks(df, signal):
    """
    Check the amount of peaks in the high resolution signal.

    Args:
        df (pd.DataFrame): dataframe with the data
        signal (str): signal to check

    Returns:
        int: minimum amount of peaks in the high resolution signal
    """
    len_h_list = []
    len_l_list = []
    for name, group in tqdm(df.groupby(['misalignment', 'recording_nr', 'direction', 'speed', 'window_id'])):
        data = group.reset_index()
        peaks_h_idx, peaks_l_idx = find_peaks(data[signal])
        len_h_list.append(len(peaks_h_idx))
        len_l_list.append(len(peaks_l_idx))
    return min(len_h_list), min(len_l_list)


def wrapped_higuchi_fd(x):
    """
    Wrapper for antropy.higuchi_fd
    """
    return ant.higuchi_fd(np.array(x))


def wrapped_ample_entropy(x):
    """
    Wrapper for antropy.sample_entropy
    """
    return ant.sample_entropy(np.array(x))


def wrapped_detrend_fluctuation(x):
    """
    Wrapper for antropy.detrended_fluctuation
    """
    return ant.detrended_fluctuation(np.array(x))


def construct_feature_collection(window_size_list):
    """
    Constructs a feature collection dictionary with the specified window sizes.

    Args:
        window_size_list (list): list of window sizes

    Returns:
        dict: feature collection dictionary
    """
    settings = {
        "spkt_welch_density": [{"coeff": c} for c in range(1,21)],
        "matrix_profile": [
            {"threshold": t, "feature": f} 
            for t in [0.9, 0.95, 0.97, 0.98, 0.99]
            for f in ["min", "max", "mean", "median", "25", "75"]
        ]

    }    

    funcs = [
        ant.perm_entropy, FuncWrapper(ant.spectral_entropy, sf=12_800), ant.svd_entropy, ant.app_entropy, wrapped_ample_entropy, 
        FuncWrapper(ant.hjorth_params, output_names=["hjorth_mobility", "hjorth_complexity"]), ant.num_zerocross, 
        ant.petrosian_fd, ant.katz_fd, wrapped_higuchi_fd, wrapped_detrend_fluctuation
    ]
    funcs.extend(tsfresh_settings_wrapper(settings))


    fc_dict = {window_size:
            FeatureCollection(
        [
            MultipleFeatureDescriptors(
                functions=funcs,
                series_names=['high_res', 'low_res'],
                windows=window_size,
                strides=window_size,
            )
        ]
    ) for window_size in window_size_list
    }
    return fc_dict


def save_features(feat_dict, data_folder):
    """
    Save features to disk

    Args:
        feat_dict (dict): Dictionary with window_size as key and feature dataframe as value
        data_folder (str): Path to data folder

    Returns:
        None
    """
    # Check if features folder exists
    features_folder = data_folder + "features/"
    if not os.path.exists(features_folder):
        # Create folder
        os.makedirs(features_folder)

    # Save features
    for window_size, feat_df in feat_dict.items():
        feat_df.to_parquet(features_folder + str(window_size) + "_features_df.parquet", index=False)
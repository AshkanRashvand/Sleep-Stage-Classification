import glob
import math
import os
import ntpath
from datetime import datetime
import numpy as np
import pandas as pd
from mne.io import read_raw_edf
from scipy.signal import butter, filtfilt  
import src.psgreader as psgreader  

# Label values
W = 0
N1 = 1
N2 = 2
N3 = 3
REM = 4
UNKNOWN = 5

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "UNKNOWN": UNKNOWN
}

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

EPOCH_SEC_SIZE = 30

# Bandpass filter parameters
LOWCUT = 0.3  # Low cutoff frequency in Hz
HIGHCUT = 35  # High cutoff frequency in Hz
FILTER_ORDER = 4

# Define a function for filtering
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Define a function for Z-score normalization
def z_score_normalization(epoch):
    mean = np.mean(epoch)
    std = np.std(epoch)
    if std == 0:
        return epoch  # Avoid division by zero; return unchanged data
    return (epoch - mean) / std

# Paths
data_dir = "/home/amyn/scratch/AIProject/data"  # Directory where your data files are located
output_dir = "/home/amyn/scratch/AIProject/output"  # Directory where outputs will be saved

# Ensure output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
select_ch = "EEG Fpz-Cz"  # EEG channel of interest

# Locate files
psg_fnames = glob.glob(os.path.join(data_dir, "*PSG.edf"))
ann_fnames = glob.glob(os.path.join(data_dir, "*Hypnogram.edf"))
psg_fnames.sort()
ann_fnames.sort()

if len(psg_fnames) != len(ann_fnames):
    raise ValueError("Mismatch between the number of PSG and Hypnogram files.")

# Process each subject
for psg_fname, ann_fname in zip(psg_fnames, ann_fnames):
    print(f"Processing files:\nPSG: {psg_fname}\nHypnogram: {ann_fname}")
    
    # Load PSG data
    raw = read_raw_edf(psg_fname, preload=True, stim_channel=None)
    sampling_rate = raw.info['sfreq']
    raw_ch_df = raw.to_data_frame()[select_ch]
    raw_ch_df = raw_ch_df.to_frame()
    raw_ch_df.set_index(np.arange(len(raw_ch_df)))

    # Apply filtering to the raw EEG signal
    raw_ch_filtered = bandpass_filter(raw_ch_df[select_ch].values, LOWCUT, HIGHCUT, sampling_rate)

    # Replace the raw signal with the filtered version
    raw_ch_df[select_ch] = raw_ch_filtered

    # Read EDF headers
    with open(psg_fname, 'r', errors='ignore') as f:
        reader_raw = psgreader.BaseEDFReader(f)
        reader_raw.read_header()
        h_raw = reader_raw.header
    raw_start_dt = datetime.strptime(h_raw['date_time'], "%Y-%m-%d %H:%M:%S")

    with open(ann_fname, 'r', errors='ignore') as f:
        reader_ann = psgreader.BaseEDFReader(f)
        reader_ann.read_header()
        h_ann = reader_ann.header
        _, _, ann = zip(*reader_ann.records())
    ann_start_dt = datetime.strptime(h_ann['date_time'], "%Y-%m-%d %H:%M:%S")

    # Verify start times
    assert raw_start_dt == ann_start_dt, "Mismatch in start times of raw and annotation files."

    # Generate labels and select labeled data
    remove_idx = []
    labels = []
    label_idx = []

    for a in ann[0]:
        onset_sec, duration_sec, ann_char = a
        ann_str = "".join(ann_char)
        label = ann2label.get(ann_str[2:-1], UNKNOWN)
        if label != UNKNOWN:
            if duration_sec % EPOCH_SEC_SIZE != 0:
                raise ValueError("Annotation duration is not a multiple of 30 seconds.")
            duration_epoch = int(duration_sec / EPOCH_SEC_SIZE)
            labels.append(np.ones(duration_epoch, dtype=int) * label)
            idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=int)
            label_idx.append(idx)
        else:
            idx = int(onset_sec * sampling_rate) + np.arange(duration_sec * sampling_rate, dtype=int)
            remove_idx.append(idx)

    labels = np.hstack(labels)

    # Remove unwanted data
    if len(remove_idx) > 0:
        remove_idx = np.hstack(remove_idx)
        select_idx = np.setdiff1d(np.arange(len(raw_ch_df)), remove_idx)
    else:
        select_idx = np.arange(len(raw_ch_df))

    label_idx = np.hstack(label_idx)
    select_idx = np.intersect1d(select_idx, label_idx)

    # Finalize raw data
    raw_ch = raw_ch_df.values[select_idx]

    if len(raw_ch) % (EPOCH_SEC_SIZE * sampling_rate) != 0:
        raise ValueError("Final raw data length is not divisible by epoch size.")
    n_epochs = len(raw_ch) // (EPOCH_SEC_SIZE * sampling_rate)

    x = np.asarray(np.split(raw_ch, n_epochs)).astype(np.float32)
    y = labels.astype(np.int32)

    # Apply Z-score normalization to each epoch
    x_normalized = np.array([z_score_normalization(epoch) for epoch in x])

    # Ensure data and labels match
    assert len(x_normalized) == len(y)

    # Save output for each subject
    filename = ntpath.basename(psg_fname).replace("-PSG.edf", ".npz")
    save_dict = {
        "x": x_normalized,  # Save the normalized data
        "y": y,
        "fs": sampling_rate,
        "ch_label": select_ch,
        "header_raw": h_raw,
        "header_annotation": h_ann,
    }
    np.savez(os.path.join(output_dir, filename), **save_dict)

    print(f"Processed and saved normalized data for {filename}")

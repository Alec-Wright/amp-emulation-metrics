import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import os
import librosa
import scipy
import pyloudnorm as pyln
import librosa

def loudnorm(db_targ, sig, sr):
    meter = pyln.Meter(sr)  # create BS.1770 meter
    ref_loud = meter.integrated_loudness(sig)
    delta_l = db_targ - ref_loud
    gain = np.power(10.0, delta_l / 20.0)
    return gain * sig



def main(sample_clips, output_loc, metrics_loc):


    test_fs = 44100

    test_name = output_loc
    clips = sample_clips

    full_db = pd.read_csv(metrics_loc)

    loud_norm = True
    targ_loud = -16

    trials = pd.read_csv(clips)

    os.makedirs(test_name, exist_ok=True)


    for i, each in enumerate(trials.iterrows()):
        os.makedirs(f"{test_name}/trial_{i}", exist_ok=True)

        clips = each[1].iloc[0:3]

        clips = full_db.iloc[clips.values]

        # Get folder name
        folder = clips['Folder Name']
        assert np.all(folder.values == folder.values[0])
        folder = folder.values[0]

        # Get clip number
        clip_num = clips['clip_id']
        assert np.all(clip_num.values == clip_num.values[0])
        clip_num = clip_num.values[0]

        # Get rate start + length
        rate_start = clips['Rate_Start'].values[0]
        rate_end = clips['Rate_End'].values[0]

        # Get ref file
        ref = f'AudioExamples/{folder}/REF-{clip_num}.wav'
        ref, fs = sf.read(ref)
        if fs != test_fs:
            ref = librosa.resample(ref, orig_sr=fs, target_sr=test_fs, axis=0)
        ref = ref[rate_start:rate_end]


        if loud_norm:
            ref = loudnorm(db_targ=targ_loud, sig=ref, sr=fs)
        #    meter = pyln.Meter(fs)  # create BS.1770 meter
        #    ref_loud = meter.integrated_loudness(ref)
        #    delta_l = targ_loud - ref_loud
        #    gain = np.power(10.0, delta_l / 20.0)
        #    ref = gain * ref

        if ref.ndim == 1:
            ref = np.array([ref, ref]).T
        sf.write(f"{test_name}/trial_{i}/ref.wav", ref, test_fs)

        for c in range(3):
            # Get clips file
            model = clips.iloc[c]["Model"]
            clippy, fs = sf.read(f'AudioExamples/{folder}/{model}-{clip_num}.wav')
            if fs != test_fs:
                clippy = librosa.resample(clippy, orig_sr=fs, target_sr=test_fs, axis=0)
            clippy = clippy[rate_start:rate_end]
            if loud_norm:
                clippy = loudnorm(db_targ=targ_loud, sig=clippy, sr=fs)
                #clippy = gain * clippy
            if clippy.ndim == 1:
                clippy = np.array([clippy, clippy]).T
            sf.write(f"{test_name}/trial_{i}/{folder}-{model}-{clip_num}.wav", clippy, test_fs)

        sos = scipy.signal.butter(4, 2000, 'lowpass', False, 'sos', 44100)
        anc2k = scipy.signal.sosfilt(sos, ref, axis=0)
        if loud_norm:
            anc2k = loudnorm(db_targ=targ_loud, sig=anc2k, sr=fs)
        sf.write(f"{test_name}/trial_{i}/{folder}-anc2k-{clip_num}.wav", anc2k, test_fs)

        #w, h = scipy.signal.freqz(b, a)

        sos = scipy.signal.butter(4, 1000, 'lowpass', False, 'sos', 44100)
        anc1k = scipy.signal.sosfilt(sos, ref, axis=0)
        if loud_norm:
            anc1k = loudnorm(db_targ=targ_loud, sig=anc1k, sr=fs)
        sf.write(f"{test_name}/trial_{i}/{folder}-anc1k-{clip_num}.wav", anc1k, test_fs)

        clips.to_csv(f"{test_name}/trial_{i}/clips.csv", index=False)
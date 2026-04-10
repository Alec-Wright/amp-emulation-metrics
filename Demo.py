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


parser = argparse.ArgumentParser()

parser.add_argument('--sample_clips', type=str, default="../WebMushraFiles/MushraSampledClips/sampled_clips_Final.csv",
                    help='file that contains csv list of sampled clips')
parser.add_argument('--output_loc', type=str, default='DemoExamples',
                    help='location where output files are saved')
parser.add_argument('--db_loc', type=str, default="../Metrics/losses_cond3_dur3_Oct15Final.csv",
                    help='location where output files are saved')

args = parser.parse_args()

if __name__ == '__main__':


    test_fs = 44100
    clips = args.sample_clips

    full_db = pd.read_csv(args.db_loc)
    trials = pd.read_csv(clips)

    for n in range(1,4):
        for clip in trials[f'C{n}']:
            print(clip)


    test_name = args.output_loc
    clips = args.sample_clips

    full_db = pd.read_csv(args.db_loc)

    loud_norm = True
    targ_loud = -16

    trials = pd.read_csv(clips)

    os.makedirs(test_name, exist_ok=True)


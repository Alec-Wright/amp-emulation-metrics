import librosa
import pyloudnorm as pyln
import numpy as np
import pandas as pd
from os.path import join, isdir


def trim_and_preproc(ref_clip, dur, top_db, fs, rfs, loud_norm):

    ref_clip = ref_clip[:, 0] if ref_clip.ndim == 2 else ref_clip
    if rfs != fs:
        ref_clip = librosa.resample(ref_clip, orig_sr=rfs, target_sr=fs)

    ref_clip, index = librosa.effects.trim(ref_clip, top_db=top_db)
    start = index[0]
    end = start + dur * fs
    ref_clip = ref_clip[0:dur * fs]

    if loud_norm:
        meter = pyln.Meter(rfs)  # create BS.1770 meter
        ref_loud = meter.integrated_loudness(ref_clip)
        delta_l = -16 - ref_loud
        gain = np.power(10.0, delta_l / 20.0)
        ref_clip = gain * ref_clip
    else:
        gain = 'na'

    if ref_clip.shape[0] < dur * fs:
        ref_clip = np.concatenate((ref_clip, np.zeros(dur * fs - ref_clip.shape[0])))

    return ref_clip, start, end, gain

def get_emb(folder, clip_num, model, feat, audio_dir, emb_dir):
    ind = pd.read_csv(join(audio_dir, folder, emb_dir, f'{feat}.csv'), index_col=0)
    embs = np.load(join(audio_dir, folder, emb_dir, f'{feat}.npy'))

    ref_ind = ind.loc[ind['0'] == join(audio_dir, folder, f'REF-{clip_num}.wav')]
    clip_ind = ind.loc[ind['0'] == join(audio_dir, folder, f'{model}-{clip_num}.wav')]

    return embs[ref_ind.index[0]], embs[clip_ind.index[0]]
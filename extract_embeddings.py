import os
import pandas as pd
import frechet_audio_distance
import numpy as np
import soundfile as sf
import librosa
from utils import misc
from tqdm import tqdm
import argparse
from os.path import join, isdir

parser = argparse.ArgumentParser(
    description='Extract audio quality metrics for all clips in AudioExamples.'
)
parser.add_argument('--audio_dir', type=str, default='FullCorpus/AudioExamples',
                    help='Path to root audio directory containing per-model subfolders')
parser.add_argument('--output_loc', type=str, default='deep_emb',
                    help='Output CSV path for computed embeddings')
parser.add_argument('--dur', type=int, default=3,
                    help='Clip duration in seconds (default: 3)')
parser.add_argument('--fs', type=int, default=44100,
                    help='Target sample rate (default: 44100)')

args = parser.parse_args()

if __name__ == '__main__':
    data = pd.read_csv(join(args.audio_dir, 'clip_metadata.csv'))

    extract_emb = True

    #for emb_name in ['vggish', 'pann', 'clap', 'encodec']:
    for emb_name in ['encodec']:
        match emb_name:
            case 'vggish':
                embedder = frechet_audio_distance.FrechetAudioDistance(verbose=False)
            case 'pann':
                embedder = frechet_audio_distance.FrechetAudioDistance(model_name='pann', sample_rate=32000)
            case 'clap':
                embedder = frechet_audio_distance.FrechetAudioDistance(model_name='clap', sample_rate=48000,
                                                                       submodel_name='630k-audioset')
            case 'encodec':
                embedder = frechet_audio_distance.FrechetAudioDistance(model_name="encodec", sample_rate=48000, channels=2)

        loud_norm = True
        dur = args.dur
        fs = args.fs

        for fn in tqdm(data['Folder Name'].unique()):
            if extract_emb:
                rows = data.loc[data['Folder Name'] == fn]
                if rows.iloc[0]['num_conds'] > 2:
                    num_clips = max(rows['clip_id'])

                    clip_names = []
                    raw_audios = []

                    for clip_num in range(1, num_clips + 1):

                        clip_names.append(f'{args.audio_dir}/{fn}/REF-{clip_num}.wav')
                        clip = rows.loc[rows['clip_id'] == clip_num]
                        ref_clip, rfs = sf.read(f'{args.audio_dir}/{fn}/REF-{clip_num}.wav')
                        ref_clip, start, end, gain = misc.trim_and_preproc(ref_clip=ref_clip, dur=3, top_db=40, fs=44100,
                                                                           rfs=rfs, loud_norm=loud_norm)
                        raw_audios.append(ref_clip)


                        for each in clip.iterrows():
                            clip_names.append(f'{args.audio_dir}/{fn}/{each[1]["Model"]}-{clip_num}.wav')
                            out_clip, rfs = sf.read(clip_names[-1])
                            out_clip = out_clip[:, 0] if out_clip.ndim == 2 else out_clip
                            if rfs != fs:
                                out_clip = librosa.resample(out_clip, orig_sr=rfs, target_sr=fs)
                            out_clip = out_clip[start:end]
                            out_clip *= gain if loud_norm else out_clip
                            if out_clip.shape[0] < dur * fs:
                                out_clip = np.concatenate((out_clip, np.zeros(dur * fs - out_clip.shape[0])))
                            raw_audios.append(out_clip)

                embeddings = embedder.get_embeddings(raw_audios, 44100)

                if emb_name == 'encodec':
                    embeddings = np.reshape(embeddings, [len(raw_audios), 128, -1])
                    embeddings = np.swapaxes(embeddings,2, 1)
                else:
                    embeddings = np.stack(embeddings, axis=0)



                os.makedirs(join(args.audio_dir, fn, args.output_loc), exist_ok=True)

                np.save(f'{args.audio_dir}/{fn}/deep_emb/{emb_name}.npy', embeddings)
                pd.DataFrame(clip_names).to_csv(f'{args.audio_dir}/{fn}/deep_emb/{emb_name}.csv')




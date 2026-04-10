import torch
from utils import losses, misc
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import librosa
from os.path import join, isdir
import argparse

def get_losses_from_ratings(items, loss_appl_, fs=44100):
    with torch.inference_mode():
        loss_csv_name = f'AudioExamples/{items["Folder Name"][0]}/aLosses.csv'
        try:
           loss_data = pd.read_csv(loss_csv_name)
        except FileNotFoundError:
           loss_data = items

        for l in loss_appl_.loss_funcs.keys():
            if l not in loss_data.columns:
                loss_data[l] = np.nan

        droplist = [i for i in loss_data.columns if i.startswith('Unnamed')]
        loss_data.drop(droplist, inplace=True, axis=1)

        for item in tqdm(loss_data.iterrows()):

            if np.isnan(np.sum(item[1][list(loss_appl_.loss_funcs.keys())].values)):

                folder, model, clip_id = item[1]['Folder Name'], item[1]['Model'], item[1]['clip_id']
                start, end = item[1]['Rate_Start'], item[1]['Rate_End']

                output_clip, rfs = sf.read(f'AudioExamples/{folder}/{model}-{clip_id}.wav')
                ref_clip, rfs = sf.read(f'AudioExamples/{folder}/REF-{clip_id}.wav')
                output_clip = output_clip[start: end]
                ref_clip = ref_clip[start:end]

                if rfs != fs:
                    output_clip = librosa.resample(output_clip, orig_sr=rfs, target_sr=fs)
                    ref_clip = librosa.resample(ref_clip, orig_sr=rfs, target_sr=fs)

                for l in loss_appl_.loss_funcs:
                    if np.isnan(item[1][l]):
                        loss_data.loc[item[0], [l]] = loss_appl_.get_loss(output_clip, ref_clip, loss_name=l)

                        loss_data.to_csv(loss_csv_name)


def get_losses_for_clip(items, loss_df, loss_appl, fs=44100, dur=3, recomp=False):
    with torch.inference_mode():

        folder = items.iloc[0]['Folder Name']
        clip_num = items.iloc[0]['clip_id']

        exerp = loss_df.loc[(loss_df['Folder Name'] == folder) & (loss_df['clip_id'] == clip_num)]

        try:
            if exerp[loss_appl.loss_funcs.keys()].isnull().values.any():
                recomp = True
        except KeyError:
            recomp = True

        if recomp:
            print(f'some losses missing {folder} - clip {clip_num}')

            ref_clip, rfs = sf.read(join(args.audio_dir,folder, f'REF-{clip_num}.wav'))
            ref_clip, start, end, gain = misc.trim_and_preproc(ref_clip=ref_clip, dur=dur, top_db=40, fs=44100, rfs=rfs,
                                                                 loud_norm=loud_norm)

            assert len(np.unique(exerp.index.values)) == len(exerp.index.values)

            for each in tqdm(exerp.iterrows()):
                if each[1][loss_appl.loss_funcs.keys()].isnull().values.any():

                    out_clip, rfs = sf.read(join(args.audio_dir, folder, f'{each[1]["Model"]}-{clip_num}.wav'))
                    out_clip = out_clip[:, 0] if out_clip.ndim == 2 else out_clip
                    if rfs != fs:
                        out_clip = librosa.resample(out_clip, orig_sr=rfs, target_sr=fs)
                    out_clip = out_clip[start:end]
                    out_clip *= gain if loud_norm else out_clip
                    if out_clip.shape[0] < dur * fs:
                        out_clip = np.concatenate((out_clip, np.zeros(dur * fs - out_clip.shape[0])))

                    assert out_clip.shape == ref_clip.shape

                    loss_df.loc[each[0], 'Rate_Start'] = start
                    loss_df.loc[each[0], 'Rate_End'] = end
                    loss_df.loc[each[0], 'Rate_Length'] = dur

                    for loss in loss_appl.loss_funcs.keys():
                        print(f'extracting {loss}')
                        if np.isnan(each[1][loss]) or recomp:

                            if type(loss_appl.loss_funcs[loss]).__name__ == 'DeepFeatLoss':
                                ref_emb, out_emb = misc.get_emb(folder=folder, clip_num=clip_num,
                                                                model=each[1].Model,
                                                                feat=loss_appl.loss_funcs[loss].feat,
                                                                audio_dir=args.audio_dir,
                                                                emb_dir=args.emb_dir)
                                new_loss = loss_appl.get_loss(out_emb, ref_emb, loss_name=loss).item()
                            else:
                                new_loss = loss_appl.get_loss(out_clip, ref_clip, loss_name=loss).item()
                            if np.isnan(new_loss):
                                print(f'nan returned by {loss}')
                            loss_df.loc[each[0], loss] = new_loss
            return loss_df

        else:
            print('losses already computed')
            return loss_df


parser = argparse.ArgumentParser(
    description='Extract audio quality metrics for all clips in AudioExamples.'
)
parser.add_argument('--audio_dir', type=str, default='FullCorpus/AudioExamples',
                    help='Path to root audio directory containing per-model subfolders')
parser.add_argument('--output_loc', type=str, default='metrics/metrics.csv',
                    help='Output CSV path for computed metrics')
parser.add_argument('--dur', type=int, default=3,
                    help='Clip duration in seconds (default: 3)')
parser.add_argument('--fs', type=int, default=44100,
                    help='Target sample rate (default: 44100)')
parser.add_argument('--emb_dir', type=str, default='deep_emb',
                    help='Target sample rate (default: 44100)')

args = parser.parse_args()


if __name__ == '__main__':
    metadata_columns = {'Folder Name': str, 'Effect Class': str, 'Effect Type': str, 'Device': str,
                        'Model': str, 'Analog/Digital/SPICE': str, 'clip_id': int, 'Rating': int,
                        'Rate_Start': int, 'Rate_End': int, 'Rate_Length': float, 'num_conds': int}

    losses_to_apply = [['VGG_l1', {'func_name': 'DeepFeatLoss', 'args': {'feat': 'VGGish', 'dist': 'l1'}}],
                       ['clap_l1', {'func_name': 'DeepFeatLoss', 'args': {'feat': 'clap', 'dist': 'l1'}}],
                       ['encodec_l1', {'func_name': 'DeepFeatLoss', 'args': {'feat': 'encodec', 'dist': 'l1'}}],
                       ['pann_l1', {'func_name': 'DeepFeatLoss', 'args': {'feat': 'pann', 'dist': 'l1'}}],

                       ['mfcc_l1', {'func_name': 'mfcc', 'args': {'dist': 'l1'}}],
                       ['mfcc_l2', {'func_name': 'mfcc', 'args': {'dist': 'l2'}}],
                       ['esr_basic', {'func_name': 'ESR'}],
                       ['mse', {'func_name': 'MSE'}],
                       ['mae', {'func_name': 'MAE'}],
                       ['bss_eval', {'func_name': 'BSSEval'}],
                       ['specconv', {'func_name': 'AuraSTFT',
                                     'args': {'w_sc': 1.0, 'w_log_mag': 0.0, 'w_lin_mag': 0.0, 'w_phs': 0.0}}],
                       ['logstft', {'func_name': 'AuraSTFT',
                                    'args': {'w_sc': 0.0, 'w_log_mag': 1.0, 'w_lin_mag': 0.0, 'w_phs': 0.0}}],
                       ['linstft', {'func_name': 'AuraSTFT',
                                    'args': {'w_sc': 0.0, 'w_log_mag': 0.0, 'w_lin_mag': 1.0, 'w_phs': 0.0}}],
                       ['mrstft', {'func_name': 'AuraMRSTFT',
                                   'args': {}}],
                       ['melstft', {'func_name': 'AuraMelSTFT',
                                   'args': {'sample_rate': 44100}}],
                       ['JTFS', {'func_name': 'JTFS', 'fs': 44100, 'dur': 3}],
                       # Package for psychloss available at https://github.com/TUIlmenauAMS/Python-Audio-Coder/tree/master
                       ['psychloss', {'func_name': 'psy_acu', 'fs': 44100}]
    ]


    loud_norm = True

    loss_appl = losses.LossApplicator()
    loss_appl.add_losses(losses_to_apply)

    all_clips = pd.read_csv(join(args.audio_dir, 'clip_metadata.csv'))

    try:
        loss_df = pd.read_csv(f'{args.output_loc}')
    except FileNotFoundError:
        loss_df = pd.DataFrame(columns=metadata_columns)
        for l in loss_appl.loss_funcs.keys():
            if l not in loss_df.columns:
                loss_df[l] = np.nan

    all_folders = sorted(set(all_clips['Folder Name']))
    for folder in tqdm(all_folders):
        rows = all_clips.loc[all_clips['Folder Name'] == folder]
        if rows.iloc[0]['num_conds'] > 2:
            num_clips = max(rows['clip_id'])
            for clip_num in range(1, num_clips+1):
                clip = rows.loc[rows['clip_id'] == clip_num]

                clips_comped = loss_df.loc[(loss_df['Folder Name'] == folder) & (loss_df['clip_id'] == clip_num)]
                if len(clips_comped) == 0:
                    loss_df = pd.concat((loss_df, clip))
                elif len(clips_comped) == len(clip):
                    pass
                else:
                    print('something gone wrong')
                    exit()
                loss_df = get_losses_for_clip(clip, loss_df, loss_appl)

                loss_df.to_csv(f'{args.output_loc}', index=False)




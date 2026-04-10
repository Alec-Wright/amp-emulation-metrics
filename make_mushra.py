#from Utils.embedding_extractor import out_clip
from utils import sample_clips, test_builder, web_mushra_maker
import argparse

parser = argparse.ArgumentParser(
    description='Create listening test and files'
)
parser.add_argument('--audio_dir', type=str, default='AudioExamples',
                    help='Path to root audio directory containing per-model subfolders')
parser.add_argument('--filename', type=str, default='metrics/metrics.csv',
                    help='Output CSV path for computed embeddings')
parser.add_argument('--sampled_clip_loc', type=str, default='sampled_clips.csv',
                    help='Output CSV path for computed embeddings')
parser.add_argument('--test_dir', type=str, default='MushraTest/MushraTest',
                    help='Output location for saved MUSHRA test')

args = parser.parse_args()

if __name__ == '__main__':

    sample_clips.main(losses_filename=args.filename, n_pages=30, sampled_clip_loc=args.sampled_clip_loc)
    test_builder.main(sample_clips=args.sampled_clip_loc, output_loc = args.test_dir, metrics_loc=args.filename)
    web_mushra_maker.main(directory=args.test_dir, output_file=args.test_dir)

    #python test_builder.py --sample_clips "MushraSampledClips/sampled_clips_Final.csv" --output_loc "webMUSHRA-master/configs/resources/audio/AudioMetricsTest" --db_loc "Metrics/losses_cond3_dur3_Oct15Final.csv"
    #python web_mushra_maker.py --direc "webMUSHRA-master/configs/resources/audio/AudioMetricsTest" --output_loc "webMUSHRA-master/configs/AudioMetricsTest"
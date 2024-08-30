import sox
import scipy.io.wavfile as wv
import pandas as pd
import tqdm

MAESTRO_DIR = '../../../../mnt/data/maestro-v3.0.0'

def uniform_sr(sr):

    maestro = pd.read_csv(f'{MAESTRO_DIR}/maestro-v3.0.0.csv', delimiter=',', quotechar='"')

    tfm = sox.Transformer()
    tfm.set_output_format(rate=sr)

    for file in maestro['audio_filename']:
        print(file)
        sample_rate, x = wv.read(f'{MAESTRO_DIR}/{file}')
        if sample_rate != sr:
            print(f'{sample_rate} -> {sr}')
            tfm.build_file(
                input_array=x,
                sample_rate_in=sample_rate,
                output_filepath=f'{MAESTRO_DIR}/{file}'
            )


if __name__ == '__main__':
    sr= 44100
    uniform_sr(sr)
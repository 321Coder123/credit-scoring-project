import os
import pandas as pd

def process_data(input_file, output):
    # Verification de l'existence du fichier input
    if not os.path.exists(input_file):
        raise FileNotFoundError('File {} not exists'.format(input_file))

    df = pd.read_csv(input_file)
    df.to_parquet(output)

    print(f'Transform terminated: {output}')


if __name__ == '__main__':
    process_data('../data/raw/application_train.csv', '../data/processed/application_train.parquet')
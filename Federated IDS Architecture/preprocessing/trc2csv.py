import argparse
from ast import parse
import pandas as pd
from pathlib import Path
from concurrent import futures
import glob

def convert_trc_to_csv(filename, dir):
    # Read the file as text file
    f = open(filename, 'r', encoding='cp949')
    content = f.read()
    f.close()

    # Process rows
    start_line = 18
    raw_data = content.split('\n')[start_line:]
    raw_data = list(map(lambda r: r.strip().split()[1:], raw_data))[:-1]

    # Convert to pandas
    columns = ['Time', 'Type', 'ID', 'Data Length']
    columns += [f'Data{x}' for x in range(8)]
    df = pd.DataFrame(raw_data, columns=columns)
    df['Time'] = df['Time'].astype('float')

    # Change the file extension to csv
    filename = '/'.join(filename.split('/')[3:])
    filename = ''.join(filename.split('.')[:-1]) + '.csv' 
    dirname = '/'.join(filename.split('/')[:-1])
    filename = '_'.join(filename.split('/')[-1].split(' '))
    savedir = Path(dir) / dirname
    savedir.mkdir(parents=True, exist_ok=True)
    
    print(f'Save to {savedir / filename}')
    df.to_csv(savedir / filename, index=False)

    return df

def convert_dir(in_dir, out_dir):
    MAX_WORKERS = 9
    files = glob.glob(in_dir, recursive=True)
    print(f'There are {len(files)} files in {in_dir}.')
    out_dirs = [out_dir for _ in range(len(files))]
    workers = min(MAX_WORKERS, len(files))
    with futures.ThreadPoolExecutor(workers) as exc:
        results = exc.map(convert_trc_to_csv, files, out_dirs)
    return len(list(results))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default=None, help='The path of file .trc')
    parser.add_argument("--out_dir", type=str, default='./datasets/csv/', help='The path of output directory')
    parser.add_argument("--in_dir", type=str, default=None, help='The path of input directory, convert all files .trc in the input directory')
    args = parser.parse_args()

    args.in_dir += '/**/*.trc'
    if args.in_dir is not None:
        num_success = convert_dir(args.in_dir, args.out_dir)
        print(f'Total number of success: {num_success}')
    elif args.f is not None:
        convert_trc_to_csv(args.f, args.out_dir)
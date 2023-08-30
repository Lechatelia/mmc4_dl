"""
Adapted from: 
    https://github.com/igorbrigadir/DownloadConceptualCaptions/blob/master/download_data.py

Requirements:
    - ImageMagick
    - See requirements.txt for python dependencies

Example Usage:
    python download_images.py --input_jsonl ./data_core/docs_no_face_shard_0_v3.jsonl
        OR
    python download_images.py --input_shards "https://storage.googleapis.com/ai2-jackh-mmc4-public/data/docs_no_face_shard_{0..23098}_v2.jsonl.zip" --output_image_dir mmc4_images/
"""

import pandas as pd
import requests
import os
import shelve
try:
    import magic
except:
    print('python-magic not installed, will not be able to detect image mimetype')
from multiprocessing import Pool
import tqdm
import argparse
import json
import subprocess
import time
import glob
from pathlib import Path
import urllib
import braceexpand
import zipfile
from PIL import Image
import random 

headers = {
    'User-Agent':'Googlebot-Image/1.0', # Pretend to be googlebot
    'X-Forwarded-For': '64.18.15.200'
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_jsonl', type=str, default=None, help='Local path to the input jsonl file')
    parser.add_argument("--input_shards", type=str, default=None, help='URL to shards')
    parser.add_argument('--output_image_dir', type=str, default=None, help='Local path to the directory that stores the downloaded images')
    parser.add_argument('--num_process', type=int, default=100, help='Number of processes in the pool can be larger than cores')
    parser.add_argument('--chunk_size', type=int, default=100, help='Number of images per chunk per process')
    parser.add_argument('--shard_name', type=str, default=None)
    parser.add_argument('--report_dir', type=str, default='./status_corev3/', help='Local path to the directory that stores the downloading status')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index of the shards')
    parser.add_argument('--end_idx', type=int, default=100, help='End index of the shards')
    
    args = parser.parse_args()

    # assert args.input_jsonl is not None or args.input_shards is not None

    # if args.input_jsonl is not None:
    #     assert args.input_jsonl.endswith('.jsonl')
        
    #     if args.shard_name is None:
    #         args.shard_name = Path(args.input_jsonl).stem
    #     print(f'Processing {args.input_jsonl} shard {args.shard_name}')
    # elif args.input_shards is not None:
    #     assert args.output_image_dir is not None
    
    # if args.output_image_dir is None:
    #     args.output_image_dir = f'./{args.shard_name}_images/'

    return args


def call(cmd):
    subprocess.call(cmd, shell=True)


def _df_split_apply(tup_arg):
    split_ind, subset, func = tup_arg
    r = subset.apply(func, axis=1)
    return (split_ind, r)


def download_images_multiprocess(args, df, func):
    """Download images with multiprocessing"""

    chunk_size = args.chunk_size
    num_process = args.num_process

    print('Generating parts...')

    shelve_filename = '%s_%s_%s_results.tmp' % (args.shard_name, func.__name__, chunk_size)
    with shelve.open(shelve_filename) as results:

        pbar = tqdm.tqdm(total=len(df), position=0)
        # Resume:
        finished_chunks = set([int(k) for k in results.keys()])
        pbar.desc = "Resuming"
        for k in results.keys():
            pbar.update(len(results[str(k)][1]))

        pool_data = ((index, df[i:i + chunk_size], func) for index, i in enumerate(range(0, len(df), chunk_size)) if index not in finished_chunks)
        pbar.write(f'\t{int(len(df) / chunk_size)} parts. Using {num_process} processes.')

        pbar.desc = f"Downloading_{args.start_idx}_{args.end_idx}"
        with Pool(num_process) as pool:
            for i, result in enumerate(pool.imap_unordered(_df_split_apply, pool_data, 2)):
                results[str(result[0])] = result
                pbar.update(len(result[1]))
        pbar.close()

    print(f'Finished downloading images for {args.input_jsonl}\nImages saved at {args.output_image_dir}')

    return shelve_filename


def _get_local_image_filename(row):
    return row['folder'] + '/' + row['local_identifier']


def download_image(row):
    fname = _get_local_image_filename(row)
    
    # print(fname)

    # Skip already downloaded images, retry others later
    if os.path.isfile(fname):
        # print(f'File {fname} already exists. Skipping...')
        row['status'] = 200
        row['file'] = fname
        # row['mimetype'] = magic.from_file(row['file'], mime=True)
        row['size'] = os.stat(row['file']).st_size
        return row

    try:
        # Use smaller timeout to skip errors, but can result in failed downloads
        response = requests.get(row['url'], stream=False, timeout=10, allow_redirects=True, headers=headers)
        row['status'] = response.status_code
        rate_limit_idx = 0
        while response.status_code == 429:
            print(f'RATE LIMIT {rate_limit_idx} for {row["local_identifier"]}, will try again in 2s')
            response = requests.get(row['url'], stream=False, timeout=10, allow_redirects=True, headers=headers)
            row['status'] = response.status_code
            rate_limit_idx += 1
            time.sleep(0.5)
            if rate_limit_idx == 5:
                print(f'Reached rate limit for {row["local_identifier"]} ({row["url"]}). Will skip this image for now.')
                row['status'] = 429
                return row

    except Exception as e:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row

    if response.ok:
        try:
            with open(fname, 'wb') as out_file:
                # some sites respond with gzip transport encoding
                response.raw.decode_content = True
                out_file.write(response.content)

            # # Resize image if it is too big
            # call('mogrify -resize "800x800>" {}'.format(fname))

            # # Use the following if mogrify doesn't exist or can't be found
            img = Image.open(fname)
            if max(img.size) > 800:
                img = img.resize((min(img.width, 800), min(img.height, 800)))
                img.save(fname)


            # row['mimetype'] = magic.from_file(fname, mime=True)
            row['size'] = os.stat(fname).st_size
        except:
            # This is if it times out during a download or decode
            row['status'] = 408
            return row
        row['file'] = fname
    return row


def save_status(args, shelve_filename):
    print(f'Generating Dataframe from results...')
    with shelve.open(shelve_filename) as results:
        keylist = sorted([int(k) for k in results.keys()])
        df = pd.concat([results[str(k)][1] for k in keylist], sort=True)
    
    report_filename = os.path.join(args.report_dir, f'{args.shard_name}.tsv.gz')
    df.to_csv(report_filename, sep='\t', compression='gzip', header=False, index=False)
    print(f'Status report saved to {report_filename}')
    
    print('Cleaning up...')
    matched_files = glob.glob(f'{shelve_filename}*')
    for fn in matched_files:
        os.remove(fn)


def gather_image_info(args):
    """Gather image info from the input jsonl"""
    data = []
    for idx in range(args.start_idx, args.end_idx):
        jsonl_file = args.input_jsonl.format(fileidx=idx)
        print(f'Processing shard {idx}')
        output_image_dir = args.output_image_dir.format(fileidx=idx)
        # skip if file not exist
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)
        if not os.path.exists(jsonl_file):
            print(f'File {jsonl_file} does not exist. Skipping...')
            continue

        with open(jsonl_file) as f:
            for line in tqdm.tqdm(f):
                info = json.loads(line.strip())
                for img_item in info['image_info']:
                    data.append({
                        'local_identifier': img_item['image_name'],
                        'url': img_item['raw_url'],
                        'folder': output_image_dir # os.path.join(jsonl_file, os.path.basename(jsonl_file).replace('.jsonl', ''))
                    })
                # break
    # with open(args.input_jsonl) as f:
    #     for line in tqdm.tqdm(f):
    #         info = json.loads(line.strip())
    #         for img_item in info['image_info']:
    #             data.append({
    #                 'local_identifier': img_item['image_name'],
    #                 'url': img_item['raw_url'],
    #             })
    return data


def gather_image_info_shard(json_file):
    """Gather image info from shard"""
    data = []
    for sample_data in tqdm.tqdm(json_file):
        # get image names from json
        sample_data = json.loads(sample_data)
        for img_item in sample_data['image_info']:
            data.append({
                'local_identifier': img_item['image_name'],
                'url': img_item['raw_url'],
            })
    return data
                

def local(args):
    # Load image info for current shard
    data = gather_image_info(args)
    # for d in data:
    #     d['folder'] = args.output_image_dir
    # import pdb; pdb.set_trace()
    df = pd.DataFrame(data)

    # import pdb; pdb.set_trace()
    # Download images
    shelve_filename = download_images_multiprocess(
        args=args, 
        df=df,
        func=download_image,
    )
    
    # Save status & cleaning up
    save_status(
        args=args,
        shelve_filename=shelve_filename,
    )

# yum install ImageMagick
# yum remove ImageMagick
# pip install python-magic

def main():
    args = parse_args()

    
    args.input_jsonl = "/apdcephfs_cq2/share_1290939/0_public_datasets/mmc4/corev3/docs_no_face_shard_{fileidx}_v3.jsonl"
    # find all files ends with .jsonl
    # using glob
 
    print(f'down file  file from {args.start_idx} to {args.end_idx}')

      
        

    args.input_shards = args.shard_name = "core3" + f"{args.start_idx}_{args.end_idx}"


    # image save path 
    args.output_image_dir = args.input_jsonl.replace("corev3", "images_corev3").replace(".jsonl", "") 

    # Prepare directory
    for _dir in [args.report_dir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    print('jsonl file path: ', args.input_jsonl)
    print('image save path: ', args.output_image_dir)

    if args.input_jsonl is not None:
        local(args=args)
    else:
        doc_shards = list(braceexpand.braceexpand(args.input_shards))

        for idx in range(len(doc_shards)):
            # image_tar = tarfile.open(image_shards[idx])
            print("Downloading zip for shard", idx)
            try:
                urllib.request.urlretrieve(doc_shards[idx], "temp.zip")

                # Open the ZIP archive and extract the JSON file
                with zipfile.ZipFile("temp.zip", "r") as zip_file:
                    # Assumes the JSON file is the first file in the archive
                    json_filename = zip_file.namelist()[0]
                    with zip_file.open(json_filename, "r") as json_file:
                        data = gather_image_info_shard(json_file)

                    shard_folder = args.output_image_dir + "/" + str(idx)
                    if not os.path.exists(shard_folder):
                        os.makedirs(shard_folder)
                    
                    for d in data:
                        d['folder'] = shard_folder

                    df = pd.DataFrame(data)

                    args.shard_name = idx

                    # Download images
                    shelve_filename = download_images_multiprocess(
                        args=args, 
                        df=df,
                        func=download_image,
                    )
                    
                    # Save status & cleaning up
                    save_status(
                        args=args,
                        shelve_filename=shelve_filename,
                    )

            except urllib.error.HTTPError as e:
                print(e)
                print("Skipping shard", idx)
                continue


if __name__ == '__main__':
    main()


    # python3 scripts/download_v4.py --start_idx 100 --end_idx 1000
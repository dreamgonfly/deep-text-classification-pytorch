from os.path import dirname, abspath, join, exists
import os
import gzip
import shutil
import argparse

from subprocess import call

import utils

BASE_DIR = dirname(abspath(__file__))
WORDVEC_DIR = join(BASE_DIR, 'wordvectors')
if not exists(WORDVEC_DIR):
    os.mkdir(WORDVEC_DIR)

def download_word2vec():
    
    # from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
    download_file = 'GoogleNews-vectors-negative300.bin.gz'
    destination = join(WORDVEC_DIR, download_file)
    print("Downloading...")
    utils.download_file_from_google_drive('0B7XkCwpI5KDYNlNUTTlSS21pQmM', destination)
    unzip_file = 'GoogleNews-vectors-negative300.bin'
    unzip_destination = join(WORDVEC_DIR, unzip_file)
    
    print("Unzipping...")
    with gzip.open(destination, 'rb') as f_in, open(unzip_destination, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
        
def download_glove():
    
    # from https://nlp.stanford.edu/projects/glove/
    dataset_url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
    download_file = 'glove.840B.300d.zip'
    download_filepath = join(WORDVEC_DIR, download_file)

    wget_command_base = "wget {dataset_url} -O {output_document}"
    wget_command = wget_command_base.format(
        dataset_url=dataset_url, output_document=download_filepath)    
    call(wget_command, shell=True)
    
    unzip_comamnd = "unzip {download_filepath} -d {wordvec_dir}".format(
        download_filepath=download_filepath, wordvec_dir=WORDVEC_DIR)
    call(unzip_comamnd, shell=True)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser("Download datasets")
    parser.add_argument("wordvec", type=str, default='word2vec', choices=['word2vec', 'glove'])
    args = parser.parse_args()

    if args.wordvec == 'word2vec':
        download_word2vec()
    elif args.wordvec == 'glove':
        download_glove()
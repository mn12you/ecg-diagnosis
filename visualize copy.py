import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
import wfdb
from tqdm import tqdm

class visualize():
    
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--record-paths', type=str, default='data/CPSC/A0010.mat', help='Path to .mat file')
        args = self.parser.parse_args()
        self.recordpaths = glob(args.record_paths)

    def plot_ecg(leads, data, title,n_cols):
        n_rows = len(leads) // n_cols
        f, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 8))
        for j in range(n_rows):
            for i in range(n_cols):
                axs[j, i].plot(data[i * n_rows + j])
                axs[j, i].spines['top'].set_visible(False)
                axs[j, i].spines['right'].set_visible(False)
                axs[j, i].set_xticks([])
                axs[j, i].set_yticks([])
                axs[j, i].set_ylabel(leads[i * n_rows + j])
                yabs_max = abs(max(axs[j, i].get_ylim(), key=abs))
                axs[j, i].set_ylim(ymin=-yabs_max, ymax=yabs_max)
        plt.savefig(f'imgs/{title}-{n_cols}.png')
        plt.close(f)

    def plot_all(self):
        for mat_file in tqdm(self.recordpaths):
            if mat_file.endswith('.mat') or mat_file.endswith('.hea'):
                mat_file = mat_file[:-4]
            patient_id = os.path.basename(mat_file)
            ecg_data, meta_data = wfdb.rdsamp(mat_file)
            leads = meta_data['sig_name']
            self.plot_ecg(leads=leads, data=ecg_data.T, title=os.path.basename(mat_file),n_cols=1)
            self.plot_ecg(leads=leads, data=ecg_data.T, title=os.path.basename(mat_file),n_cols=2)

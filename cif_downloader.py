import json
import os
import pandas as pd
import re
import urllib.request
from tqdm import tqdm
import shutil


class CIFDownloader:
    def __init__(self, json_file, csv_file, cifs_folder='cifs'):
        self.json_file = json_file
        self.csv_file = csv_file
        self.cifs_folder = cifs_folder
        self.summaries = None
        self.data = None

    def read_json_file(self):
        with open(self.json_file) as f:
            summaries = json.load(f)
            temp = {val: key for key, val in summaries.items()}
            self.summaries = {val: key for key, val in temp.items()}

    def filter_csv_data(self):
        self.data = pd.read_csv(self.csv_file)
        self.data = self.data[self.data.doi.isin(list(self.summaries.keys()))]
        self.data = self.data[self.data.doi.notnull()]
        self.data = self.data[self.data.doi != '']

    def add_summaries_to_data(self):
        self.data['summary'] = self.data['doi'].map(self.summaries)
        self.data = self.data[self.data.summary.notnull()]
        self.data = self.data[self.data.summary != '']

    def clean_summaries(self):
        def clean(data):
            p = re.compile(r'<.*?>')
            return p.sub('', data)
        self.data['summary'] = self.data['summary'].apply(clean)

    def create_folder(self):
        if not os.path.exists(self.cifs_folder):
            os.makedirs(self.cifs_folder)

    def download_cif_files(self):
        self.data = self.data.reset_index()
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            urllib.request.urlretrieve(
                f"http://www.crystallography.net/cod/{row['file']}.cif",
                f"{self.cifs_folder}/{row['file']}.cif"
            )

    def save_dataframe(self):
        self.data[['file', 'summary']].to_csv(
            os.path.join(self.cifs_folder, "id_prop.csv"),
            index=False,
            header=False
        )

    def copy_json_file(self, src_file, dst_folder=None):
        if dst_folder is None:
            dst_folder = self.cifs_folder
        shutil.copyfile(src_file, os.path.join(dst_folder, 'atom_init.json'))

    def run(self):
        self.read_json_file()
        self.create_folder()
        self.filter_csv_data()
        self.add_summaries_to_data()
        self.clean_summaries()
        self.download_cif_files()
        self.save_dataframe()


# Usage of the CIFDownloader class
if __name__ == "__main__":
    downloader = CIFDownloader(
        json_file='summaries.json', csv_file='cod_new.csv')
    downloader.run()  # Execute all the steps
    # Copy atom_init.json to cifs folder
    downloader.copy_json_file('atom_init.json')

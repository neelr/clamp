import os
import json
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer


class CSVDownloader:
    def __init__(self, url, destination):
        self.url = url
        self.destination = destination

    def download(self):
        response = requests.get(self.url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        with tqdm(total=total_size, unit='iB', unit_scale=True) as t:
            with open(self.destination, 'wb') as file:
                for data in response.iter_content(block_size):
                    t.update(len(data))
                    file.write(data)
            if total_size != 0 and t.n != total_size:
                print("ERROR, something went wrong")


class PackageInstaller:
    def __init__(self, packages):
        self.packages = packages

    def install(self):
        for package in self.packages:
            os.system(f"pip install {package}")


class PDFDownloader:
    def __init__(self, uri):
        self.uri = uri

    def download_and_read(self):
        response = requests.get(self.uri)
        return response.content


class DocumentSummaryGenerator:
    def __init__(self, model_name, tokenizer_name, device='mps'):
        self.tokenizer = PegasusTokenizer.from_pretrained(
            tokenizer_name, max_position_embeddings=2048)
        self.model = PegasusForConditionalGeneration.from_pretrained(
            model_name, max_position_embeddings=2048).to(device)
        self.device = device

    def generate_summary(self, src_text):
        # Assuming 'src_text' is a list of documents.
        batch = self.tokenizer(
            src_text, truncation=True, padding="longest", return_tensors="pt").to(self.device)
        translated = self.model.generate(**batch)
        return self.tokenizer.batch_decode(translated, skip_special_tokens=True)


class DOIAbstractExtractor:
    def __init__(self, doi):
        self.doi = doi

    def get_abstract(self):
        try:
            response = requests.get(
                f"https://doi.org/{self.doi}", headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.content, 'html.parser')
            abstr = soup.find('meta', attrs={'name': 'DCTERMS.abstract'})
            if not abstr:
                abstr = soup.find('meta', attrs={'name': 'citation_abstract'})
            if abstr:
                return abstr["content"]
        except Exception as e:
            print(f"An error occurred: {e}")
            return None


class SummaryCollector:
    def __init__(self, csv_file, output_file, doi_column='doi'):
        self.csv_file = csv_file
        self.output_file = output_file
        self.doi_column = doi_column

    def collect_summaries(self, summary_generator, num_samples=None):
        data = pd.read_csv(self.csv_file)
        doi_sum = {}

        for i, row in tqdm(data.iterrows(), total=data.shape[0]):
            if num_samples is not None and i >= num_samples:
                break

            doi_text = DOIAbstractExtractor(
                row[self.doi_column]).get_abstract()
            if doi_text:
                summary = summary_generator.generate_summary([doi_text])[0]
                doi_sum[row[self.doi_column]] = summary

            # Save progress every 10 iterations
            if i % 10 == 0:
                with open(self.output_file, 'w') as fp:
                    json.dump(doi_sum, fp)

        # Final save of the summaries
        with open(self.output_file, 'w') as fp:
            json.dump(doi_sum, fp)

# Example Usage:


# Download CSV file
downloader = CSVDownloader(
    "http://www.crystallography.net/cod/result.php?format=csv", "cod.csv")
downloader.download()

# Install required packages
installer = PackageInstaller(
    ["scidownl", "PyPDF2", "transformers", "torch", "sentencepiece"])
installer.install()

# Generate summaries
summary_generator = DocumentSummaryGenerator(
    "google/pegasus-cnn_dailymail", "google/pegasus-cnn_dailymail")
collector = SummaryCollector("cod_new.csv", "summaries_inorganic.json")
collector.collect_summaries(summary_generator, num_samples=50)

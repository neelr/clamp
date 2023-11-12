# 🌟 CLAMP: Contrastive Language And Molecule Pre-training Network 🌟

<img width="433" alt="image" src="https://github.com/neelr/clamp/assets/35831013/640a9c12-7add-443d-b56b-ec01782fa4ce" style="align-self:center" >


Welcome to the official repository of CLAMP (Contrastive Language And Molecule Pre-training Network) - the frontier in marrying the power of language with the intricacies of molecular structures to revolutionize material generation! 🔬🧬

## What is CLAMP? 🤔

CLAMP is a cutting-edge deep learning framework designed to foster innovation in material science. At its heart, CLAMP is an architecture that understands and generates materials by leveraging the massive untapped potential of textual data paired with crystal structures, gathered from the vast expanses of open-source research papers. 📚✨

Through a fusion of a Convolutional Graph Neural Network (CGCNN) encoder and a language encoder, we breathe life into a contrastive model that is not only incredibly smart but also fun to work with - who knew molecules and words could be this cool together?

## Abstract 📝
https://s.neelr.dev/clamp-research-paper

This paper highlights a shift in how to approach material generation. Instead of material-to-material, we propose a language-to-material generation architecture that utilizes millions of untapped data points. Using a web scraper to collect crystal text pairs from open-source research papers, a contrastive model can be trained using a convolutional graph neural network encoder and a language encoder. This would allow unsupervised zero-shot classification which can be trained by taking advantage of linguistic structure. Without any specific training data, an ~82\% accuracy was achieved and ~75\% accuracy for photocatalyst prediction with an extremely small dataset. This novel network could ideally be cross-applied to any reaction that can be described via text, opening completely new methods to think about 3D chemical framework generation. In the full experiment diffusion models would likely be incorporated to fully exploit the latent space.

## Features of CLAMP 🌈

- **Here for your zero-shot classification needs:** Achieve unsupervised classification on completely new data with impressive accuracy - no specific training required!
- **Web-scraping:** Integrates easily with web-scraping tools to collate crystal structure and text data pairs directly from research papers. Knowledge is power, and CLAMP harnesses it from the web!
- **Language to material:** Utilizes the structure and vastness of linguistic data to predict and generate material properties. It's like having a super-smart alchemist at your fingertips!
- **Photocatalyst prediction:** With just a small dataset, CLAMP dazzles with its ability to predict photocatalyst effectiveness. 🌞⚗️
- **Future-forward:** Built with diffusion models in mind for later experiments, it's a springboard for exploring the uncharted territories of latent chemical space!

## How to Get Started? 💼

Here's a simple guide to kick off your adventure with CLAMP:

### Step 1: Clone the Repository 🏗️

```sh
git clone https://github.com/neelr/clamp.git
cd clamp
```

### Step 2: Install Dependencies 🛠

Make sure you've got all the ingredients for this alchemical experiment!

```sh
pip install -r requirements.txt
```

### Step 3: Set Up Your Dataset 📊

Download the dataset from https://www.kaggle.com/datasets/programgeek01/cif-summary-data or compile it with `cif_downloader.py` and `annotation_scraper.py`

### Step 4: Train Your Model 🏋️‍

Train CLAMP with your data to become the sorcerer supreme of material science!

```python
python clamp_model.py
```

### Step 5: Unleash the Power 🔮

Use your trained CLAMP model to predict material properties or generate new crystals. Watch as the magic unfolds!

## Prerequisites 🗝

- Python 3.6+
- PyTorch
- A sprinkle of creativity
- A dash of enthusiasm for science!

## Contributing 🤝

Jump in and join the alchemy party! Whether it's adding new features, fixing bugs, or spreading the word, your contributions are what make CLAMP not just a code base, but a community.

## License 📜

CLAMP is open source, because we believe in the power of sharing knowledge. It's licensed under MIT, so feel free to use it, modify it, and innovate with it.

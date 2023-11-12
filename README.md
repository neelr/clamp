# 🌟 CLAMP: Contrastive Language And Molecule Pre-training Network 🌟

<img width="433" alt="image" src="https://github.com/neelr/clamp/assets/35831013/640a9c12-7add-443d-b56b-ec01782fa4ce" style="align-self:center" >

## Abstract 📝
https://s.neelr.dev/clamp-research-paper

This paper highlights a shift in how to approach material generation. Instead of material-to-material, we propose a language-to-material generation architecture that utilizes millions of untapped data points. Using a web scraper to collect crystal text pairs from open-source research papers, a contrastive model can be trained using a convolutional graph neural network encoder and a language encoder. This would allow unsupervised zero-shot classification which can be trained by taking advantage of linguistic structure. Without any specific training data, an ~82\% accuracy was achieved and ~75\% accuracy for photocatalyst prediction with an extremely small dataset. This novel network could ideally be cross-applied to any reaction that can be described via text, opening completely new methods to think about 3D chemical framework generation. In the full experiment diffusion models would likely be incorporated to fully exploit the latent space.

## Features of CLAMP 🌈

- **Zero-shot classification:** Achieve unsupervised classification on completely new data with ~82% accuracy-no specific training required!
- **First web-scraped crystal-text dataset:** The files autoscrape cif-text pairs from the internet, collecting ~222k crystal-text pairs which is unheard of at the moment.
- **Language to material:** Utilizes the structure and vastness of linguistic data to predict and generate material properties.
- **Photocatalyst prediction:** With just a small dataset, CLAMP proves its ability with a 75% accuracy in photocatalyst prediction, showing its ability.
- **Future Steps:** Built with diffusion models in mind for later experiments, it's a springboard for exploring the latent chemical space!

## How to Get Started? 💼

Here's a simple guide to kick off dev with CLAMP:

### Step 1: Clone the Repo 🏗️

```sh
git clone https://github.com/neelr/clamp.git
cd clamp
```

### Step 2: Install Dependencies 🛠

```sh
pip install -r requirements.txt
```

### Step 3: Set Up Your Dataset 📊

Download the dataset from https://www.kaggle.com/datasets/programgeek01/cif-summary-data or compile it with `cif_downloader.py` and `annotation_scraper.py`

### Step 4: Train Your Model 🏋️‍

Train CLAMP with your data

```python
python clamp_model.py
```

### Step 5: Unleash the Power 🔮

Use your trained CLAMP model to predict material properties or generate new crystals.

## Prerequisites 🗝

- Python 3.6+
- PyTorch

## Contributing 🤝

Jump in and join the party! Whether it's adding new features, fixing bugs, or spreading the word, your contributions are what make CLAMP not just a code base, but a community. Hopefully it becomes as widespread as CLIP is for image generation.

## License 📜

CLAMP is open source, because we believe in the power of sharing knowledge. It's licensed under MIT, so feel free to use it, modify it, and innovate with it.

# Non-Agentic AI/ML Training Directions — Research Digest

*A survey of domains where transformers, GNNs, diffusion models, SSMs, and KANs are pushing scientific frontiers — outside of agent/tool-use systems like L.O.R.A./CodeOn. Compiled August 2026.*

## Why this list

Every domain below follows the AlphaFold pattern: take a neural architecture, point it at a well-defined scientific prediction or generation problem with a clear ground truth (a structure, a forecast, a signal), and train. No planning loop, no tool use, no multi-step autonomy — just perception, prediction, or generation. Several map directly onto architecture families you already study, which is flagged per section.

## Quick reference

| Domain | Core architecture | Overlaps your study |
|---|---|---|
| 1. Protein structure & design | Axial attention (Evoformer) + diffusion | Diffusion, transformers |
| 2. Genomics / regulatory DNA | StripedHyena (SSM+conv hybrid) | **SSMs/Mamba**, byte-level models |
| 3. Materials discovery | GNN (message passing) + diffusion | **GNNs**, diffusion |
| 4. Weather/climate | GNN (icosahedral mesh) + diffusion | **GNNs**, diffusion |
| 5. Chemistry (reactions) | Seq2seq transformer over SMILES | Transformers |
| 6. Physics simulation (PDE) | Fourier Neural Operator, transformer | Transformers, KAN-adjacent |
| 7. Particle physics | Set transformer / GNN on point clouds | **GNNs**, transformers |
| 8. Astronomy | Multimodal masked transformer | Transformers |
| 9. Neuroscience / brain decoding | Contrastive alignment + diffusion decoder | Diffusion, transformers |
| 10. World models | JEPA (self-supervised latent prediction) | **JEPA/world models** (direct) |
| 11. Time-series forecasting | Decoder transformer / tokenized-as-language | Transformers, SSM-adjacent |
| 12. Remote sensing | ViT + masked autoencoding | Transformers |
| 13. Symbolic regression | KAN (learnable spline edges) | **KANs** (direct) |

(bold = direct hit on architecture families you actively study)

---

## 1. Structural biology — protein structure & design

**What it is:** predicting or generating the 3D shape of biomolecules, and how they bind each other, directly from sequence.

**Flagship systems:**
- **AlphaFold 2/3** (Google DeepMind + Isomorphic Labs) — AF2 uses the Evoformer (attention over multiple-sequence-alignments plus pairwise residue representations); AF3 replaced the structure module with a diffusion model and extended prediction to protein–DNA/RNA/ligand complexes.
- **ESM3** (EvolutionaryScale) — a 98B-parameter generative model treating sequence, structure, and function as three token tracks fused in one transformer, trained on 2.78B natural protein sequences. Used to design esmGFP, a fluorescent protein ~500M "evolutionary years" from anything natural.
- **Isomorphic Labs' IsoDDE** (Feb 2026) — a proprietary successor pushing protein-ligand accuracy well past AF3, already in early clinical drug trials.

**Data:**
- RCSB Protein Data Bank — rcsb.org — canonical experimentally-solved structures, free bulk download.
- AlphaFold Protein Structure Database — alphafold.ebi.ac.uk — 200M+ predicted structures.
- UniProt — uniprot.org — sequence/annotation database AlphaFold DB is keyed against.
- AlphaFold3 code/weights for academic use, and ESM3-open weights — both on GitHub/Hugging Face.

**Scaled-down project:** fine-tune ESM2/ESM3-open, or build a small Evoformer-lite, on a narrow under-studied protein family (e.g. a specific extremophile or plant protein class).

---

## 2. Genomics — regulatory DNA/RNA foundation models

**What it is:** reading and writing the genome — predicting what a mutation does, or generating novel functional sequences.

**Flagship systems:**
- **Evo / Evo2** (Arc Institute, Stanford, NVIDIA) — trained on 9.3 trillion nucleotides across all three domains of life at up to 40B parameters. Built on **StripedHyena 2**, which hybridizes convolutional/gated operators with attention — the team's own scaling-law study found plain transformers scale worse at single-nucleotide, byte-level resolution.
- **AlphaGenome** (Google DeepMind, *Nature*, Jan 2026) — takes 1 megabase of DNA and predicts thousands of regulatory tracks (splicing, chromatin accessibility, expression) at single-base resolution, building on the earlier Enformer model.
- **Bonus — mechanistic interpretability angle:** Goodfire trained sparse autoencoders on Evo2's internal layers and found biologically meaningful features (α-helices, tRNAs) emerging — a ready-made mech-interp project if that appeals.

**Data:**
- OpenGenome / OpenGenome2 (Arc Institute) — the Evo/Evo2 pretraining corpora, on Hugging Face.
- ENCODE (encodeproject.org) and GTEx (gtexportal.org) — regulatory/expression validation data.
- Code: `arcinstitute/evo2` and `google-deepmind/alphagenome` on GitHub (AlphaGenome API free for non-commercial use).

**Scaled-down project:** fine-tune Evo (7B) or train a small StripedHyena yourself on one under-studied genome (a virus family, an extremophile, a crop pathogen) and benchmark against a same-size plain transformer — a clean "does the SSM-hybrid inductive bias help here" result. This is your most direct SSM bridge on the whole list.

---

## 3. Materials science — crystal discovery & generative design

**What it is:** predicting which never-synthesized atomic combinations form stable, useful crystals — batteries, solar cells, superconductors.

**Flagship systems:**
- **GNoME** (Google DeepMind, *Nature* 2023) — a GNN trained on DFT data plus active learning, proposed 2.2M candidate materials, 380,000+ confirmed stable, roughly sextupling humanity's known stable-crystal count.
- **MatterGen** (Microsoft Research, *Nature* 2025/2026) — inverts the problem: a diffusion model generating a crystal (atom positions, elements, lattice) conditioned on a target property. A generated material (TaCr₂O₆) was synthesized; its measured bulk modulus landed within 20% of the design target.

**Data:**
- Materials Project — materialsproject.org — the core open DFT database (150k+ materials), free API, and the base dataset both models build on.
- GNoME's released dataset (380k+ structures) — `google-deepmind/materials_discovery` on GitHub, Apache-licensed.
- MatterGen code + training data — `microsoft/mattergen` on GitHub and Hugging Face.
- Open Quantum Materials Database — oqmd.org — an independent DFT database.

**Scaled-down project:** train a small property-conditioned diffusion model restricted to one chemical family (e.g. one battery-cathode candidate class) — tractable because the DFT labeling work is already done for you.

---

## 4. Weather & climate forecasting

**What it is:** forecasting atmospheric evolution directly from data, without solving fluid-dynamics equations at inference time.

**Flagship systems:**
- **GraphCast** (DeepMind, *Science* 2023) — GNN-based deterministic forecaster, beat ECMWF's best physics-based system (HRES) on most variables and lead times.
- **GenCast** (DeepMind, *Nature* 2024) — a diffusion model generating 50+ member probabilistic ensembles, beating ECMWF's ENS ensemble on 97%+ of tested variable/lead-time combinations.
- **WeatherNext 2** (2026) — current operational successor unifying both lines, plus a cyclone-tracking variant.

**Data:**
- ERA5 reanalysis — Copernicus Climate Data Store, cds.climate.copernicus.eu — the 40+ year dataset all of these train on, free with registration.
- Model weights + code: `google-deepmind/weathernext` on GitHub; outputs also queryable via Google Cloud BigQuery/Earth Engine and the open OpenMeteo API.

**Scaled-down project:** a regional model — train a small GNN on ERA5 restricted to South Asia / the Indian monsoon (a well-studied but still-imperfect forecasting problem) — instead of attempting a global retrain.

---

## 5. Chemistry — reaction prediction & retrosynthesis

**What it is:** given a target molecule, predict what reaction produced it (retrosynthesis) or what a reaction yields (forward prediction) — chemistry as translation.

**Flagship systems:**
- Transformer seq2seq models (Molecular Transformer, Chemformer, RSGPT, ReactionT5) treat reactions as translation over SMILES strings.
- **RetroRules 2026** — an expanded template database unifying biochemical (enzymatic) and organic (USPTO-derived) reaction templates.

**Data:**
- USPTO-50k / USPTO-full — the standard benchmark, ~50k–1M+ reactions from US patents, widely mirrored on Hugging Face/GitHub.
- RetroRules (via its API) and the Open Reaction Database (open-reaction-database.org) for broader reaction sources.
- PubChem (pubchem.ncbi.nlm.nih.gov) and ChEMBL (ebi.ac.uk/chembl) for molecule-level data.

**Scaled-down project:** fine-tune a small pretrained seq2seq model (e.g. ReactionT5) on a reaction class USPTO underrepresents, or reproduce a retrosynthesis transformer from scratch on USPTO-50k. Probably the gentlest on-ramp on this whole list — no new architecture family, just a chemical tokenizer.

---

## 6. Physics simulation — neural PDE surrogates

**What it is:** learning to approximate PDE solutions (fluid flow, plasma, elasticity) orders of magnitude faster than a numerical solver.

**Flagship systems:**
- **Fourier Neural Operator (FNO)** and **DeepONet** — learn operators (function-to-function mappings) rather than fixed-size-vector mappings, so a trained model generalizes across resolution and boundary conditions.
- **The Well** (Polymathic AI, 2024–2026) — 15TB, 16+ curated simulation datasets (fluid dynamics, MHD supernova explosions, acoustic scattering, biological pattern formation), released to give this field an "ImageNet moment," pretrained FNO baselines included.
- **Walrus** (Polymathic AI, 2026) — a foundation model trained across all of the Well's fluid-like domains at once, showing real cross-domain transfer (e.g. stellar physics to bacterial swimming).

**Data:**
- The Well — `github.com/PolymathicAI/the_well`, streamable from Hugging Face (`polymathic-ai` org), unified PyTorch loader, pretrained FNO checkpoints included as baselines.

**Scaled-down project:** pick one of the Well's smaller sub-datasets and train your own operator against the provided pretrained baseline. Close to the most turnkey entry point on this whole list — dataset curation and baselines are already done.

---

## 7. Particle physics — collider/detector data

**What it is:** classifying the particle spray ("jets") from LHC collisions to identify which fundamental particle produced it.

**Flagship systems:**
- **ParticleNet** — treats a jet as an unordered point cloud, dynamic graph CNN.
- **Particle Transformer (ParT)** — augments attention with physics-derived pairwise particle features, near state-of-the-art for jet tagging.

**Data:**
- CERN Open Data Portal — opendata.cern.ch — real ATLAS/CMS collision data and simulation, free.
- HLS4ML LHC jet dataset — ~880k labeled jets, 5 classes, commonly hosted on Zenodo (search "hls4ml LHC jet dataset Zenodo"), built specifically for benchmarking compact/fast taggers.
- Top Quark Tagging Reference dataset (Kasieczka et al.) — another standard public benchmark, also on Zenodo.

**Scaled-down project:** implement ParT and a state-space/linear-attention alternative on HLS4ML and compare accuracy vs. inference latency — genuinely useful, since the field's real bottleneck is real-time trigger latency (40 MHz collision rate), not just accuracy. Single-GPU feasible.

---

## 8. Astronomy — multimodal survey foundation models

**What it is:** a shared representation across very different telescope data types (images, spectra, catalogs) so the model can infer one modality from another.

**Flagship system:**
- **AION-1** (Polymathic AI, 2025–2026) — 300M–3.1B-parameter transformer trained on 200M+ observations across five major surveys (Legacy Survey, HSC, SDSS, DESI, Gaia), per-modality tokenizers feeding one shared masked-modeling transformer over 39 data types.

**Data:**
- AION-1 code + weights — `github.com/PolymathicAI/AION`, Hugging Face (`polymathic-ai` org).
- The underlying "Multimodal Universe" collection (100TB+ ML-ready astronomical data), linked from the AION repo.
- Individual survey archives: SDSS (sdss.org), the ESA Gaia archive, DESI public data releases.

**Scaled-down project:** freeze AION-1's encoder and train a lightweight probe for a task it wasn't directly evaluated on — a rare-object classifier, or an instrument-specific calibration task. "Linear probe on a frozen foundation model" is cheap and still a real result.

---

## 9. Neuroscience — brain signal decoding

**What it is:** reconstructing what someone saw, heard, or is thinking about, directly from fMRI/EEG/MEG.

**Flagship systems:**
- A now-large family of fMRI-to-image pipelines (Brain-Diffuser, MindEye-style) pairing a learned brain-signal encoder with a frozen pretrained image diffusion model, using the brain signal to condition generation.
- EEG-based variants (e.g. BReAD) do the same with cheaper, more accessible EEG/MEG via retrieval-augmented diffusion.

**Data:**
- Natural Scenes Dataset (NSD) — naturalscenesdataset.org — large 7T fMRI-while-viewing-images dataset, the standard fMRI benchmark here.
- THINGS-data / THINGS-EEG2 (Gifford et al.) — object-recognition EEG+fMRI+behavior dataset, much lighter-weight than fMRI.
- OpenNeuro — openneuro.org — the general open repository hosting most public fMRI/EEG/MEG datasets, including the above.
- Human Connectome Project — humanconnectome.org — large-scale connectivity data for a network-analysis angle instead.

**Scaled-down project:** EEG is the tractable entry point — build an EEG-to-CLIP-embedding encoder on THINGS-EEG2 and decode via a frozen pretrained diffusion model. Single-GPU feasible, well-defined baselines to beat. Arguably the domain here closest to your Landscape-of-Consciousness work.

---

## 10. World models — self-supervised physical prediction

**What it is:** learning how the physical world evolves purely by watching video, without ever generating pixels.

**Flagship systems:**
- **V-JEPA 2** (Meta, 2025) — 1.2B-parameter encoder pretrained on 1M+ hours of internet video via masked *latent*-prediction (no pixel decoder, no text, no labels), then post-trained on under 62 hours of robot video for zero-shot pick-and-place planning.
- **Genie 3** (Google DeepMind, public since Jan 2026) — generates explorable, promptable 3D worlds in real time from a text/image prompt; coherent for a few minutes at 720p/24fps. Closed model, no public weights — inspiration/comparison only, not something you can train on.

**Data:**
- V-JEPA 2 code + weights — `github.com/facebookresearch/vjepa2`, open PyTorch.
- Something-Something v2 and Epic-Kitchens-100 — the standard action-understanding benchmarks V-JEPA2 is evaluated on.

**Scaled-down project:** reproduce a mini-JEPA (image or short-video version) on a small controlled dataset and test whether latent-space prediction gives better sample efficiency than a pixel-reconstruction baseline downstream. This is the direct production-scale version of the JEPA idea you already study — V-JEPA2's whole pitch (predict in latent space, don't reconstruct pixels) is the cleanest real-world demonstration of the thesis at scale.

---

## 11. Time-series foundation models

**What it is:** zero-shot forecasting for numerical series (energy demand, sensors, epidemiological curves) without retraining per series.

**Flagship systems:**
- **TimesFM** (Google Research) — ~200M-parameter decoder-only transformer, patch-based, currently ranked #1 on the GIFT-Eval benchmark across 28 datasets.
- **Chronos / Chronos-2** (Amazon) — tokenizes numeric values via scaling+quantization and reuses a T5-style language model, treating a series as a "sentence" of value-tokens.
- **Moirai** (Salesforce) — trained on a 27-billion-observation corpus (LOTSA) for cross-frequency, any-variate forecasting.

**Data:**
- Code: `google-research/timesfm`, `amazon-science/chronos-forecasting` on GitHub.
- Monash Time Series Forecasting Archive — forecastingdata.org — dozens of standard benchmark series.
- GIFT-Eval and standard M4/M5 competition datasets for evaluation.

**Scaled-down project:** train a small Mamba-backbone forecaster using Chronos's tokenization scheme and benchmark against Chronos-2/TimesFM zero-shot on a domain-specific series (e.g. local traffic or air-quality data if sourceable) — a direct "SSM vs. tokenized-LM" result on a concrete task.

---

## 12. Earth observation / remote sensing

**What it is:** general-purpose satellite-imagery representations that transfer to flood mapping, crop yield, deforestation tracking, disaster response with very little labeled data.

**Flagship systems:**
- **Prithvi-EO-2.0** (NASA + IBM + Jülich Supercomputing Centre) — 600M-parameter ViT-based masked autoencoder trained on Harmonized Landsat-Sentinel-2 data; recently became the first geospatial foundation model deployed in orbit (a South Australian satellite and the ISS).
- **Clay v1.5** — an open, Apache-licensed alternative.
- **AlphaEarth Foundations** (Google DeepMind) — an embedding-field model for global mapping from sparse labels, accessible via Google Earth Engine.

**Data:**
- Prithvi weights — huggingface.co (`ibm-nasa-geospatial` org). Clay — clay-foundation.github.io/model.
- Harmonized Landsat-Sentinel-2 (HLS) data via NASA Earthdata (earthdata.nasa.gov).
- Google Earth Engine — earthengine.google.com — free petabyte-scale imagery for research/nonprofit use.

**Scaled-down project:** fine-tune Prithvi or Clay on a labeled task for a specific Indian region — flood extent, crop type, or urban heat-island mapping around Bengaluru — using a few hundred to a few thousand labels instead of the tens of thousands a from-scratch model needs. This is exactly the use case these models were built around.

---

## 13. Symbolic regression & scientific law discovery — your most direct architecture match

**What it is:** recovering an actual closed-form equation from data instead of a black-box predictor — the AI-Feynman problem.

**Flagship systems / recent work:**
- KANs were explicitly pitched around this use case (Liu et al. 2024; the newer *Phys. Rev. X* "KANs Meet Science" paper) — because each edge has a learnable 1D spline instead of a fixed nonlinearity, a trained KAN can often be read off directly as a formula. Demonstrated recovering conserved quantities, Lagrangians, symmetries, and constitutive laws.
- **KAN-SR** (2025) — divide-and-conquer symbolic regression built on KANs, recovers ground-truth equations from the Feynman-SRSD benchmark.
- **Symbolic-KAN** (2026) — extends this into inverse dynamical systems and physics-informed PDE settings.
- **PySR** (Miles Cranmer) remains the strongest non-KAN baseline across most published SRSD comparisons.

**Data:**
- SRSD-Feynman benchmark (120+120 datasets, Feynman-Lectures-derived formulas, with dummy-variable robustness tests) — `github.com/omron-sinicx/srsd-benchmark`.
- PySR — `github.com/MilesCranmer/PySR` — also a strong baseline to beat.
- Original KAN repo (`pykan`, Liu et al.).
- The Well (#6) doubles as a much harder, non-benchmark testbed once SRSD starts to feel easy.

**Scaled-down project:** the most self-contained project on this whole list — no scraping, no API keys, runs on a laptop. Reproduce KAN-SR on SRSD-Feynman, then push onto something genuinely under-explored: a constitutive law from a slice of the Well's fluid data, or a governing equation from one of your own grokking-experiment setups — see whether the KAN can name what the network learned.

---

## Bonus, explicitly flagged as agent-adjacent (not really "non-agentic")

**AlphaEvolve** (DeepMind, 2025–2026) and its predecessor **FunSearch** pair an LLM with an evolutionary loop and an automated evaluator to discover new algorithms — most famously beating Strassen's 1969 bound on 4×4 complex matrix multiplication. DeepMind's own paper describes AlphaEvolve as an evolutionary *coding agent* — it's a close cousin of what you're already building with L.O.R.A./CodeOn (LLM-orchestrated, tool-using, iterative), not a departure from it. Included here for completeness, but it doesn't really satisfy the "not agentic" part of your ask — skip it unless the algorithm-discovery angle specifically appeals.

---

## If you want the shortlist: most tractable for a solo/small-team project

Given you're mid-undergrad, likely Colab/Kaggle-class compute, and aiming at a genuine contribution rather than a class project, these four stand out:

1. **KAN-SR extended past SRSD onto real spatiotemporal data (#6/#13).** Cheapest to run, most directly extends what you already study. "Can a KAN read out a governing equation from genuinely messy data, not a clean textbook benchmark" is a real question worth checking against current literature before committing.
2. **SSM/Mamba-backbone vs. attention on particle jet tagging (#7).** Small, clean, well-benchmarked dataset (HLS4ML), a real motivation (trigger-latency constraints), and a crisp SSM-vs-attention comparison — same flavor of question your SSM interest already points toward, applied to a concrete domain instead of a toy task.
3. **EEG brain decoding on THINGS-EEG2 (#9).** Personally resonant given your consciousness-corpus work, single-GPU feasible, and the diffusion-decoding pattern is well-documented to build on.
4. **Fine-tune a released foundation model on a locally-relevant task (#2 or #12).** Evo2 on a specific organism, or Prithvi/Clay on a Bengaluru-region Earth-observation task. Lowest technical risk — the expensive pretraining is already done, so your contribution is the application, evaluation, and domain-specific adaptation.

Do a quick literature check (arXiv, Google Scholar, Semantic Scholar) before committing to any of these — what's flagged as open territory reflects what surfaced in research just now, and this space moves fast enough that something may have shipped in the last few months that closes a gap.

## Compute notes

- Kaggle gives 30 free GPU-hours/week (T4/P100); Colab free/Pro tiers cover most of the scaled-down projects above.
- Almost every flagship model listed has pretrained checkpoints on Hugging Face — fine-tuning or probing a released model is nearly always more tractable than pretraining from scratch, and still counts as a real research contribution if the downstream task or domain is novel.

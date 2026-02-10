# CONSISTENT TEXT-TO-IMAGE GENERATION VIA SCENE DE-CONTEXTUALIZATION (SDeC)

Official codebase for the ICLR 2026 paper “Consistent Text-to-Image Generation via Scene De-Contextualization” (OpenReview: https://openreview.net/forum?id=rRp8yYKRGj).

---

## ✨ Highlights
- Consistent multi-frame generation with identity prompts and frame prompt sequences.
- Scene de-contextualization via a trainable SVD amplifier that decouples ID from scene context.
- SDXL-compatible pipeline built on Stable Diffusion XL and `diffusers`.

---

## 📌 Paper
- **Title**: CONSISTENT TEXT-TO-IMAGE GENERATION VIA SCENE DE-CONTEXTUALIZATION
- **Venue**: ICLR 2026
- **Authors**: Song Tang, Peihao Gong, Kunyu Li, Kai Guo, Boyu Wang, Mao Ye, Jianwei Zhang, Xiatian Zhu
- **Project Page**: *TBD*
- **OpenReview**: https://openreview.net/forum?id=rRp8yYKRGj
- **PDF**: *TBD*

**Abstract**
> Consistent text-to-image (T2I) generation seeks to produce identity-preserving images of the same subject across diverse scenes, yet it often fails due to a phenomenon called identity (ID) shift. Previous methods have tackled this issue, but typically rely on the unrealistic assumption of knowing all target scenes in advance. This paper reveals that a key source of ID shift is the native correlation between subject and scene context, called scene contextualization, which arises naturally as T2I models fit the training distribution of vast natural images. We formally prove the near-universality of this scene-subject correlation and derive theoretical bounds on its strength. On this basis, we propose a novel, efficient, training-free prompt embedding editing approach, called Scene De-Contextualization (SDeC), that imposes an inversion process of T2I’s built-in scene contextualization. Specifically, it identifies and suppresses the latent scene-subject correlation within the ID prompt’s embedding by quantifying SVD directional stability to re-weight the corresponding eigenvalues adaptively. Critically, SDeC allows for per-scene use (one prompt per scene) without requiring prior access to all target scenes. This makes it a highly flexible and general solution well-suited to real-world applications where such prior knowledge is often unavailable or varies over time. Experiments demonstrate that SDeC significantly enhances identity preservation while maintaining scene diversity.

---

## 🚀 Quick Start
```
$ conda create --name SDeC python=3.10
$ conda activate SDeC
$ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 
$ pip install transformers==4.46.3  # or: conda install conda-forge::transformers 
$ conda install -c conda-forge diffusers
$ pip install opencv-python scipy gradio==4.44.1 sympy==1.13.1
$ pip install compel
### Install dependencies ENDs ###

# Run infer code
$ python main.py
```

---

## 🔒 License
MIT/Apache-2.0

---

## 📎 Citation

```bibtex
@inproceedings{Tang2026SDeC,
  title     = {Consistent Text-to-Image Generation via Scene De-Contextualization},
  author    = {Tang, Song and Gong, Peihao and Li, Kunyu and Guo, Kai and Wang, Boyu and Ye, Mao and Zhang, Jianwei and Zhu, Xiatian},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://openreview.net/forum?id=rRp8yYKRGj}
}
```

---

## 🙏 Acknowledgements
Built upon:
- Hugging Face `diffusers`
- `compel`
- Stable Diffusion XL

---
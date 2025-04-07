# Enhancing NR-IQA Model Robustness through Simple Image Compression Techniques

This is the official code for our TCSVT paper: "Enhancing NR-IQA Model Robustness through Simple Image Compression Techniques." [Paper Link] In this implementation, we use one-step FGSM as the attack method and HyperIQA as the target NR-IQA model as an example. The proposed JPEG and JPEG+NT defense strategies can be applied to other NR-IQA models and attack methods in a similar manner. Meanwhile, the Norm regularization Training (NT) strategy is introduced in [this work](https://github.com/YangiD/DefenseIQA-NT).

---

## Directory Structure

```
project/
│
├── checkpoints/         # Pretrained models (you need to download manually)
├── test_data/           # Dataset folder
├── livec-test.csv       # Data information (filename and MOS)
├── models.py            # HyperIQA architectures
├── hyperIQAclass.py     # HyperIQA model
├── FGSM...py            # FGSM attack against HyperIQA with compressed adversarial examples saved
├── test_performance.py  # Test attack performance with and without JPEG compression
└── requirements.txt     # Environment dependencies
```

---

## Requirements

The following packages are required:

- Python
- PyTorch
- Torchvision
- Pillow
- NumPy
- SciPy

You can also install required packages via pip:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1. Prepare Data

Place your dataset in the test_data/ directory, and prepare the livec-test.csv file containing the filenames and their corresponding MOS values.

---

### 2. Download Pretrained Checkpoints

Download pretrained models from [Google Drive](https://drive.google.com/drive/folders/1QcfvPJbMhyGiTnh4Evt5VmQwBbJEy5pD?usp=drive_link)

Then **place the downloaded `.pth` file(s)** into the `checkpoints/` directory:

```
checkpoints/
├── livec_bs16_wo_nt.pth
└── livec_bs16_nt_0.001.pth
```

---

### 3. Attack the NR-IQA model

To attack the normally trained HyperIQA model, run:

```bash
python FGSM_HyperIQA_compression.py --nt_weight 0 --jpeg_com 70
```

To attack the HyperIQA model trained with Norm regularization (HyperIQA-NT), run:

```bash
python FGSM_HyperIQA_compression.py --nt_weight 0.001 --jpeg_com 70
```

---

### 4. Evaluate

To evaluate the attack performance with JPEG defense, run:

```bash
python test_performance.py --nt_weight 0
```

To evaluate the attack performance with JPEG+NT defense, run:

```bash
python test_performance.py --nt_weight 0.001
```

---

## Citation

If you use this code, please cite our paper:

```bibtex
@journal{yujia_TCSVT_2025,
  title={Enhancing NR-IQA Model Robustness through Simple Image Compression Techniques},
  author={Liu, Yujia and Yang, Chenxi and Yu, Zhaofei and Huang, Tiejun},
  booktitle={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025}
}
```

---

## Contact

If you have any questions or suggestions, feel free to reach out:

**Yujia Liu**  
[📧 yujia_liu@pku.edu.cn] 

---
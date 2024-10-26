
<p align="center">
  <img src="dwnet.png">
</p>

```
conda create -n dn python=3.11
conda activate dn
pip install -r requirements.txt

copy dnetccnl_doc3d.pkl and unetnc_doc3d.pkl to eval/models

python infer.py --img_path=./eval/inp/4_2.png --out_path=4_2.png
```

# DewarpNet
This repository contains the codes for [**DewarpNet**](https://www3.cs.stonybrook.edu/~cvl/projects/dewarpnet/storage/paper.pdf) training.

### Recent Updates
- **[May, 2020]** Added evaluation images and an important note about Matlab SSIM.
- **[Dec, 2020]** Added OCR evaluation details.
- **[Sep, 2021]** Released DewarpNet final models used in the paper.

### Training
- Prepare Data: `train.txt` & `val.txt`. Contents should be like:
```
1/824_8-cp_Page_0503-7Ns0001
1/824_1-cp_Page_0504-2Cw0001
```
- Train Shape Network:
`python trainwc.py --arch unetnc --data_path ./data/DewarpNet/doc3d/ --batch_size 50 --tboard`
- Train Texture Mapping Network:
`python trainbm.py --arch dnetccnl --img_rows 128 --img_cols 128 --img_norm --n_epoch 250 --batch_size 50 --l_rate 0.0001 --tboard --data_path ./DewarpNet/doc3d`

### Inference:
- Run:
`python infer.py --wc_model_path ./eval/models/unetnc_doc3d.pkl --bm_model_path ./eval/models/dnetccnl_doc3d.pkl --show`

### Evaluation (Image Metrics):
- We use the same evaluation code as [DocUNet](https://www3.cs.stonybrook.edu/~cvl/docunet.html).
To reproduce the quantitative results reported in the paper use the images available [here](https://drive.google.com/drive/folders/1OOcChuWphGJ22PC_vAL2DV872ssWwdRX?usp=share_link).

- **[Important note about Matlab version]** We noticed that Matlab 2020a uses a different SSIM implementation which gives a better MS-SSIM score (0.5623). Whereas we have used Matlab 2018b. Please compare the scores according to your Matlab version.

### Evaluation (OCR Metrics):
- The 25 images used for OCR evaluation is ```/eval/ocr_eval/ocr_files.txt```
- The corresponding ground-truth text is given in ```/eval/ocr_eval/tess_gt.json```
- For the OCR errors reported in the paper we had used cv2.blur as pre-processing which gives higher error in all the cases. For convenience, we provide the updated numbers (without using blur) in the following table:

|      Method      |    ED   |       CER      | ED  (no blur) | CER (no blur) |
|:----------------:|:-------:|:--------------:|:-------------:|:-------------:|
|      DocUNet     | 1975.86 |  0.4656(0.263) |    1671.80    | 0.403 (0.256) |
| DocUNet on Doc3D | 1684.34 | 0.3955 (0.272) |    1296.00    | 0.294 (0.235) |
|     DewarpNet    | 1288.60 | 0.3136 (0.248) |    1007.28    | 0.249 (0.236) |
|  DewarpNet (ref) | 1114.40 | 0.2692 (0.234) |     812.48    | 0.204 (0.228) |
- We had used the Tesseract (v4.1.0) default configuration for evaluation with PyTesseract (v0.2.6).

### Models:
- Pre-trained models are available [here](https://drive.google.com/file/d/114NfUhxlf_XV0uV7ZTdJTUxME0cW0Ty9/view?usp=share_link). These models are captured prior to  end-to-end training, thus won't give you the end-to-end results reported in Table 2 of the paper. Use the images provided above to get the exact numbers as Table 2.
- Final models are available [here](https://drive.google.com/drive/folders/1yFiYBIkrY61IuRniiV4MLF3jyrNeVd2I?usp=sharing). These models can be used to unwarp DocUNet images and **reproduce the results in the ICCV paper**.

### Dataset:
- The *doc3D dataset* can be downloaded using the scripts [here](https://github.com/cvlab-stonybrook/doc3D-dataset).

### More Stuff:
- [Demo](https://sagniklp.github.io/dewarpnet-demo/)
- [Project Page](https://www3.cs.stonybrook.edu/~cvl/projects/dewarpnet/)
- [Doc3D Rendering Codes](https://github.com/sagniklp/doc3D-renderer)
### Citation:
If you use the dataset or this code, please consider citing our work-
```
@inproceedings{SagnikKeICCV2019,
Author = {Sagnik Das*, Ke Ma*, Zhixin Shu, Dimitris Samaras, Roy Shilkrot},
Booktitle = {Proceedings of International Conference on Computer Vision},
Title = {DewarpNet: Single-Image Document Unwarping With Stacked 3D and 2D Regression Networks},
Year = {2019}}
```
#### Acknowledgements:
- These codes are heavily structured on [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg).

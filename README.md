<div align="center">
  <p>
    <a align="center" target="_blank">
      <img width="100%" src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/banner/YOLO_fastest_banner.png"></a>
  </p>

</div>

# <div align="center">NetsPresso tutorial for YOLO Fastest compression</div>
## Order of the tutorial
[0. Sign up](#0-sign-up) </br>
[1. Install](#1-install) </br>
[2. Prepare dataset](#2-prepare-dataset) </br>
[3. Training](#3-training) </br>
[4. Compress model, convert to tflite, and benchmark with PyNetsPresso](#4-compress-model-convert-to-tflite-and-benchmark-with-pynetspresso) </br>
</br>


## 0. Sign up
To get started with the NetsPresso Python package, you will need to sign up either at <a href="https://netspresso.ai?utm_source=git_yolo&utm_medium=text_np&utm_campaign=py_launch" target="_blank">NetsPresso</a> or <a href="https://py.netspresso.ai/?utm_source=git_yolo&utm_medium=text_py&utm_campaign=py_launch" target="_blank">PyNetsPresso</a>.
</br>

## 1. Install
Clone repo and install [requirements.txt](https://github.com/Nota-NetsPresso/ModelZoo-YOLOFastest-for-ARM-U55-M85/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch >= 1.11, < 2.0**](https://pytorch.org/get-started/locally/).
```bash
git clone https://github.com/Nota-NetsPresso/ModelZoo-YOLOFastest-for-ARM-U55-M85.git  # clone
cd ModelZoo-YOLOFastest-for-ARM-U55-M85
pip install -r requirements.txt  # install
```
</br>

## 2. Prepare dataset
Download the STREETS dataset and annotations from [link](https://databank.illinois.edu/datafiles/ht2io/download), unzip, and move the vehicleannotaitons folder to ../dataset/ directory

```
Your code structure should like

├── datasets
│    └── vehicleannotaitons
│         ├── images
│         └── annotations
│    
└── ModelZoo-YOLOFastest-for-ARM-U55-M85

```
</br>

## 3. Training
If you want to start from scratch, create a '.pt' file via 'train.py'.
```bash
python train.py --data ./data/STREETS.yaml --epochs 300 --weights '' --cfg ./models/yolo-fastest.yaml  --batch-size 64
```
</br>

## 4. Compress model, convert to tflite, and benchmark with PyNetsPresso

`auto_process.py` provides integrated process which contains torch.fx converting, model compression, fx model retraining, onnx exporting, tflite converting, device benchmark, and mAP validation. You can execute `auto_process.py` with minimal training hyper-parameters and NetsPresso account information.

You can choose Renesas-RA8D1 (Arm Cortex-M85) or Ensemble-E7-DevKit-Gen2 (Arm Cortex-M55 + Ethos-U55) device, and boost inference speed by giving Helium option.
``` bash
python auto_process.py --data ./data/STREETS.yaml --name yolo_fastest --weight_path ./models/yolo_fastest_streets.pt --epochs 300 --batch-size 64 --np_email '' --np_password '' --target_device Renesas-RA8D1 --helium
```
</br>

## Benchmark

|Model                                                                                           | Format|Precision| Size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | Speed<br><sup>Cortex-M85<br>(ms) | Speed<br><sup>Cortex-M85 with Helium<br>(ms) | Speed<br><sup>Ethos-U55<br>(ms) | Params<br><sup>(M) |
|-------|-----|-----|--------------------- | -------------------- | ----------------- | ------------ | ----------------------------- | ------------------------------ | ------------------ |
| [YOLO-Fastest](https://github.com/Nota-NetsPresso/ModelZoo-YOLOFastest-for-ARM-U55-M85/tree/master/models/yolo_fastest_uadetrac_256.pt)           | PyTorch |FP32   | 256                   | 41.6                | 75.5              | **-**                       | **-**                       | **-**                        | **0.3**            |
[YOLO-Fastest](https://github.com/Nota-NetsPresso/ModelZoo-YOLOFastest-for-ARM-U55-M85/tree/master/models/yolo_fastest_streets_full_int8_256.tflite)     | TFLite|Full INT8         | 256              | 39.7           | 72.8              | **594**                       | **269**                       | **6.8**                        | **0.3**            |
Compressed YOLO-Fastest    |TFLite | Full INT8         | 256              | 37.3           | 71.5              | **513**                       | **234**                       | **6.0**                        | **0.2**            |
<details>
  <summary>Table Notes</summary>

- The checkpoint is trained to 300 epochs with default settings. The model uses [hyp.scratch-low.yaml](https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml) hyps.
- **mAP<sup>val</sup>** values are for single-model single-scale on the [STREETS](https://databank.illinois.edu/datasets/IDB-3671567) dataset.<br>Reproduce by `python val.py --weights './models/yolo_fastest_streets_256.pt' --data ./data/STREETS.yaml --img 256` for pytorch ckpt file, and
 `python val.py --weights './models/yolo_fastest_streets_full_int8_256.tflite' --data ./data/STREETS.yaml --img 256 --anchors-for-tflite-path ./models/yolo_fastest_streets_256_anchors.json` for full int8 tflite file.
- **Speed** is making inference for a STREETS val image using Cortex-M85 (with/without helium) and Ethos-U55.<br>

</details>

## <div align="center">Contact</div>

Join our <a href="https://github.com/orgs/Nota-NetsPresso/discussions">Discussion Forum</a> for providing feedback or sharing your use cases, and if you want to talk more with Nota, please contact us <a href="https://www.nota.ai/contact-us">here</a>.</br>
Or you can also do it via email(contact@nota.ai) or phone(+82 2-555-8659)!

<div align="center">
  <p>
    <a align="center" target="_blank">
      <img width="100%" src="https://github.com/Nota-NetsPresso/NetsPresso-Compatible-Models/blob/main/imgs/banner/YOLOv5_banner.png"></a>
  </p>

</div>

# <div align="center">NetsPresso tutorial for YOLO Fastest compression</div>
## Order of the tutorial
[0. Sign up](#0-sign-up) </br>
[1. Install](#1-install) </br>
[2. Training](#2-training) </br>
[3. Compress model and export to onnx with PyNetsPresso](#3-compress-model-and-export-to-onnx-with-pynetspresso) </br>
[4. Benchmark](#4-benchmark) </br>
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

## 2. Training
If you want to start from scratch, create a '.pt' file via 'train.py'.
```bash
python train.py --data coco.yaml --epochs 300 --weights '' --cfg ./models/yolo-fastest.yaml  --batch-size 32
```
</br>

## 3. Compress model and export to onnx with PyNetsPresso

`auto_process.py` provides integrated process which contains torch.fx converting, model compression, fx model retraining, and onnx exporting. You can execute `auto_process.py` with minimal training hyper-parameters and NetsPresso account information.

``` bash
python auto_process.py --data coco.yaml --name yolo_fastest --weight_path yolo_fastest_uadetrac_4jh.pt --epochs 300 --batch-size 128 --np_email '' --np_password ''
```
</br>

## 4. Benchmark

|Model                                                                                           | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | mAP<sup>val<br>50 | Speed<br><sup>Cortex-M85<br>(ms) | Speed<br><sup>Cortex-M85 with helium<br>(ms) | Speed<br><sup>Ethos-U55<br>(ms) | params<br><sup>(M) |
| ----------------------------------------------------------------------------------------------- | --------------------- | -------------------- | ----------------- | ---------------------------- | ----------------------------- | ------------------------------ | ------------------ |
| [YOLO-Fastest](https://github.com/Nota-NetsPresso/ModelZoo-YOLOFastest-for-ARM-U55-M85/tree/master/models/yolo_fastest_uadetrac_256.pt)              | 256                   | 24.0                 | 43.2              | **593**                       | **253**                       | **6.7**                        | **0.3**            |

## <div align="center">Contact</div>

Join our <a href="https://github.com/orgs/Nota-NetsPresso/discussions">Discussion Forum</a> for providing feedback or sharing your use cases, and if you want to talk more with Nota, please contact us <a href="https://www.nota.ai/contact-us">here</a>.</br>
Or you can also do it via email(contact@nota.ai) or phone(+82 2-555-8659)!

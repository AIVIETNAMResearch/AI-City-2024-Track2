# [CVPRW 2024] Divide and Conquer Boosting for Enhanced Traffic Safety Description and Analysis with Large Vision Language Model 

[Khai Trinh Xuan](https://github.com/trinhxuankhai), [Khoi Nguyen Nguyen](https://github.com/nguyen-brat), [Bach Hoang Ngo](https://github.com/BachNgoH), [Vu Dinh Xuan](https://github.com/dxv2k), [Minh-Hung An](https://github.com/anminhhung), Quang-Vinh Dinh

⭐ The 2nd Place Solution to The 8th NVIDIA AI City Challenge (2024) Track 2 from AIO_ISC Team.

[![Read the Paper](https://img.shields.io/badge/Paper-red)](https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Xuan_Divide_and_Conquer_Boosting_for_Enhanced_Traffic_Safety_Description_and_CVPRW_2024_paper.pdf)

<p align="center">
    <img src="figures/main_figure.jpg"/>
</p>

---
## Results

| **Rank**            |       **Team ID**       |         **Team name**          |             **MRR Score**              |
|:--------------------:|:-----------------------------:|:----------------------------:|:---------------------------------:|
| 1 |   208   |   AliOpenTrek   |   33.4308    |
| **2** |   **28**  | **AIO_ISC (Ours)** |   **32.8877**   |
| 3 |   68  |      Lighthouse       |   32.3006    |
| 4 |   87 |    VAI     |   32.2778    |
| 5 |   184  |    Santa Claude	    |   29.7838    |

---

## Structure
### Code Structure
```
├── src
│   ├── preprocess
│   |   ├── extract_frames
│   |   ├── segment_extraction
│   ├── train
│   |   ├── Qwen-VL
│   |   ├── prepare_train_data
│   ├── inference
│   ├── postprocess
│   ├── evaluation
├── tools
├── aux_dataset
│   ├── results
│   ├── submission
│   ├── train_data
│   ├── extracted_frames
│   ├── segmentation_data
├── dataset
```

### Data Structure
Please download [WTS dataset](https://github.com/woven-visionai/wts-dataset) and set up the dataset as follow:
```
├── dataset
│   ├── annotations
│   |   ├── bbox_annotated
│   |   |   ├── pedestrian
│   |   |   |   ├── train
│   |   |   |   ├── val
│   |   |   |   ├── test
│   |   |   ├── vehicle
│   |   ├── bbox_generated
│   |   |   ├── ... (same structure)
│   |   ├── caption
│   |   |   |   ├── train
│   |   |   |   ├── val
│   |   |   |   ├── test
│   |   videos
│   |   |   ├── train
│   |   |   ├── val
│   |   |   ├── test
│   |   external
│   |   |   ├── BDD_PC_5K
│   |   |   |   ├── annotations
|   |   |   │   |   ├── bbox_annotated
|   |   │   |   |   |   ├── train
|   |   │   |   |   |   ├── val
|   |   │   |   |   |   ├── test
|   |   |   │   |   ├── bbox_generated
|   |   │   |   |   ├── ... (same structure)
|   |   |   │   |   ├── caption
|   |   │   |   |   ├── ... (same structure)
│   |   |   |   ├── videos
│   |   |   |   |   ├── train
│   |   |   |   |   ├── val
│   |   |   |   |   ├── test
```

## Environment
```
pip install -r requirements.txt
```

## Prepare
Run the following instructions to create our final submission or you can download our aux_dataset (not including extracted_frames) folder [here](https://drive.google.com/file/d/1o5E1c8ePIW6HtMcVy72PQmrU3z1Mzffb/view?usp=sharing). 
Inorder to run the repo, the user should use Nvidia GPU which has ampere architecture (rtx 3000 series, A5000, A6000,...) or higher (Hopper, Blackwell).
### Preprocessing
Extracting video frames:
```
sh tools/extract_frames.sh
```

Segment Extraction: please create a new environment follow [here](src/preprocess/segment_extraction/README.md) and run:
```
sh tools/segment_extraction.sh
```
Remember to grant permisson to access [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) model on the huggingface hub.

### Training
Prepare train data:
```
sh tools/prepare_train_data.sh
```

Training:
Set the correct train and eval data path and run the code in [here](src/train/Qwen-VL/finetune/finetune_lora_single_gpu.sh).

### Inference
Inference trained model on test set:
```
sh tools/inference.sh
```
The pretrained checkpoints uploaded to huggingface hub are listed in [here](src/inference/ckpt.json). 

### Postprocessing
```
sh tools/postprocess.sh
```
After run postprocessing, you can submit the file [aux_dataset/submission.json](aux_dataset/submission.json) to the official evaluation server.

### Evaluation
We follow [wts-dataset repo](https://github.com/woven-visionai/wts-dataset) and reimplement the fast version at [here](src/evaluation/metrics.py).

## Docker
We provide Dockerfile to build Segment Extraction environment :
```
sudo docker build -t segment_extraction .
sudo docker run -it --gpus all -v ./:/home/code -w /home/code segment_extraction
```
## Citation
If you have any questions, please leave an issue or contact us: trinhxuankhai2310@gmail.com
```
@InProceedings{Xuan_2024_CVPR,
    author    = {Xuan, Khai Trinh and Nguyen, Khoi Nguyen and Ngo, Bach Hoang and Xuan, Vu Dinh and An, Minh-Hung and Dinh, Quang-Vinh},
    title     = {Divide and Conquer Boosting for Enhanced Traffic Safety Description and Analysis with Large Vision Language Model},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {7046-7055}
}
```

## Acknowledgement
Our VLM training code relies on [Qwen-VL](https://github.com/QwenLM/Qwen-VL) repo. Thanks for their great work!

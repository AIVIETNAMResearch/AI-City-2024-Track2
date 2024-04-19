# AICITY2024 Track 2 - Code from AIO_ISC Team
## Structure
### Code Structure
```
├── src
│   ├── preprocess
│   |   ├── extract_frames
│   |   ├── segment_extraction
│   ├── train
│   |   ├── prepare_train_data
│   |   ├── Qwen-VL
│   ├── inference
│   ├── postprocess
│   ├── evaluation
├── aux_dataset
│   ├── extracted_frames
│   ├── segmentation_data
│   ├── train_data
│   ├── results
├── dataset
```

### Data Structure
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
│   |   |   |   ├── ... (same structure)
```

## Environment
### Preprocessing + postprocessing
```
pip install -r requirements.txt
```
### Training
Please set up the environment following [Qwen-VL environment](src/train/Qwen-VL/README.md).

## Prepare
### Preprocessing
Extracting video frames :
```
sh tools/extract_frames.sh
```
Segment Extraction :
```
Updating
```

### Training
Prepare train data :
```
Updating
```
Training :

Set the correct train and eval data path and run the code in [here](src/train/Qwen-VL/finetune/finetune_lora_single_gpu.sh).

### Inference
Updating.

### Postprocessing
Updating.

### Evaluation
We follow [wts-dataset repo](https://github.com/woven-visionai/wts-dataset) and reimplement the fast version at [here](src/evaluation/metrics.py).


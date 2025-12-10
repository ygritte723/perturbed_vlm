import os

# Base paths - Update these to match your environment
IMAGES_ROOT = "/jet/home/lisun/work/xinliu/images"

# Found in: evaluation/image_classification/test_clip.py
CHEXPERT_DATA_DIR = os.path.join(IMAGES_ROOT, "CheXpert-v1.0-small")
CHECKPOINT_PATH = "/jet/home/lisun/work/xinliu/hi-ml/hi-ml-multimodal/src/caches-wopretrain/bt64/cache-2023-11-27-22-06-16-moco/model_last.pth"

# Found in: scripts/clip.py
INDIANA_DATA_DIR = "/ocean/projects/asc170022p/lisun/xinliu/images"
INDIANA_CAPTIONS_CSV = os.path.join(INDIANA_DATA_DIR, "csv/indiana_captions.csv")
INDIANA_IMAGES_NORMALIZED = os.path.join(INDIANA_DATA_DIR, "images_normalized")

# Derived paths for CheXpert
CHEXPERT_TRAIN_CSV = os.path.join(CHEXPERT_DATA_DIR, "train_mod1.csv")
CHEXPERT_VALID_CSV = os.path.join(CHEXPERT_DATA_DIR, "valid_mod.csv")
CHEXPERT_TEST_CSV = os.path.join(CHEXPERT_DATA_DIR, "test_mod.csv")

# Checkpoint paths
CHECKPOINT_PATH_OUR = "/jet/home/lisun/work/xinliu/hi-ml/hi-ml-multimodal/src/new_caches_v7/T0.1_L0.1_shuffle-temp0.01/cache-2023-11-27-06-06-56-moco/model_last.pth"
CHECKPOINT_PATH_GLORIA = "/jet/home/lisun/work/xinliu/gloria/caches/wopretrain/bt12/cache-2023-11-27-02-28-55-moco/model_last.pth"

# Output directories
OUTPUT_DIR = "output"

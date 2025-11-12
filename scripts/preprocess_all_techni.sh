# CUDA_VISIBLE_DEVICES='0' python scripts/pre_technicolor.py --videopath $1/Birthday --startframe 151 --endframe 201
# CUDA_VISIBLE_DEVICES='0' python scripts/pre_technicolor.py --videopath $1/Fabien   --startframe  51 --endframe 101
# CUDA_VISIBLE_DEVICES='0' python scripts/pre_technicolor.py --videopath $1/Painter  --startframe 100 --endframe 150
# CUDA_VISIBLE_DEVICES='0' python scripts/pre_technicolor.py --videopath $1/Theater  --startframe  51 --endframe 101
# CUDA_VISIBLE_DEVICES='0' python scripts/pre_technicolor.py --videopath $1/Train    --startframe 151 --endframe 201

# exp1: colmap_00001 ~ colmap_00251
CUDA_VISIBLE_DEVICES=3 python scripts/pre_technicolor.py --videopath /mnt/nvme1/semyu/data/dynamic/unsync/exp1/Technicolor/Undistorted/Birthday/dataset_colmap --startframe 1 --endframe 281
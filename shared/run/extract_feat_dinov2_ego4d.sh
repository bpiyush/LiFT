# DATA_DIR="/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap"
# csv="${DATA_DIR}/ego4d_chiral_subset-v1-with_reverse_captions-560K_>=0.5s_buffer=0.2.csv"
# video_dir="${DATA_DIR}/cut_full_scale"
dataset='ego4d_subset'
si=$1
ei=$2

python adapt4change/scripts/compute_dino_features.py \
    --dataset $dataset \
    --no_filter_chiral \
    --si $si \
    --ei $ei
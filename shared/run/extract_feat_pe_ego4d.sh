# csv="/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap/metadata/cleaned_chiral_subset_with_reverse_captions-425K-2025-07-17_18:21:38.csv"
# csv="/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap/metadata/cleaned_chiral+general-850K-2025-07-17_18:21:38.csv"
csv="/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap/metadata/cleaned_chiral+general-816K-2025-07-17_18:21:38.csv"
output_dir="/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap/features/"
text_col="caption_forward"
id_col="id"
si=$1
ei=$2

python chiral_retrieval/scripts/compute_video_text_features.py \
--csv $csv  \
--output_dir $output_dir \
--text_col $text_col \
--id_col $id_col \
--si $si \
--ei $ei \
--devices 1
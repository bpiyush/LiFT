video_dir=/scratch/shared/beegfs/shared-datasets/EGO4D/ego4d_data_v1/full_scale/
cut_dir=/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap/cut_full_scale/
# csv=/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap/ego4d_chiral_subset-v1-with_reverse_captions-650K.csv
# csv="/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap/ego4d_chiral_subset-v1-with_reverse_captions-560K_>=0.5s_buffer=0.2.csv"
# csv="/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap/ego4d_chiral_subset-v1-with_reverse_captions-490K_>=0.5s_buffer=0.2.csv"
csv="/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap/metadata/cleaned_chiral+general-850K-2025-07-17_18:21:38.csv"
si=$1
ei=$2

echo "CSV: $csv"
echo "Start index: $si"
echo "End index: $ei"
echo "--------------------------------"

# First, cut the original videos into clips based on the CSV
echo "Cutting clips..."
python shared/scripts/cut_clips_fast.py \
    --csv $csv \
    --video_dir $video_dir \
    --cut_dir $cut_dir \
    --video_id_key video_id \
    --start_time_key start_sec \
    --end_time_key stop_sec \
    --si $si \
    --ei $ei

echo "Done cutting clips."
echo "--------------------------------"

# # Then, downsize the videos to 360 width
# echo "Downsizing videos..."
# python shared/scripts/downsize_videos_simple.py --video_dir $cut_dir --remove_old --width 360
# echo "Done downsizing videos."
# echo "--------------------------------"


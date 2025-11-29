CUDA_VISIBLE_DEVICES=0 bash shared/run/extract_feat_dinov2_ego4d.sh 0 110000 &
CUDA_VISIBLE_DEVICES=1 bash shared/run/extract_feat_dinov2_ego4d.sh 110000 220000 &
CUDA_VISIBLE_DEVICES=2 bash shared/run/extract_feat_dinov2_ego4d.sh 220000 330000 &
CUDA_VISIBLE_DEVICES=3 bash shared/run/extract_feat_dinov2_ego4d.sh 330000 440000 & wait
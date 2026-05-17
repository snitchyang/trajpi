python -m debugpy --listen 5678 --wait-for-client scripts/verify_replica_lerobot_actions.py \
  --lerobot-root=datasets/mshab/mshab_lerobot \
  --rearrange-dataset-root=mshab/dataset/scene_datasets/replica_cad_dataset/rearrange-dataset \
  --num-episodes=10000 --frames-per-episode=3 --seed=0
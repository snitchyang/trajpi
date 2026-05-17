python -m debugpy --listen 0.0.0.0:5678 --wait-for-client scripts/serve_policy.py --port 8001 policy:checkpoint \
    --policy.config=pi05_mstraj \
    --policy.dir=/data/user/wzhang834/users/vick/trace_mobile/openpi/outputs/checkpoints/pi05_mstraj/mshab_traj_action_20260425_122526/2500 \

DATA_ROOT=""
NUM_ITER=1200
STEP=35


SUBJECT="0011_1"
DATA_FOLDER="0011_02_1_w2a"

# SUBJECT="0011_2"
# DATA_FOLDER="0011_02_2_w2a"

# SUBJECT="0013"
# DATA_FOLDER="0013_02_w2a"

# SUBJECT="0038"
# DATA_FOLDER="0038_04_w2a"

# SUBJECT="0039"
# DATA_FOLDER="0039_02_w2a"

# SUBJECT="0041"
# DATA_FOLDER="0041_00_w2a"

STAGE1_RENDERING=output/ocmotion/${SUBJECT}/train/${NUM_ITER}/renders/
python3 stage2_oc.py --subject $DATA_FOLDER --data_root ${DATA_ROOT}/${DATA_FOLDER} --stage1_render_path $STAGE1_RENDERING --num_inference_steps $STEP # --overwrite
DATA_ROOT=""
NUM_ITER=1800
OPTIM_NUM_ITER=1200


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

CKPT=output/ocmotion/${SUBJECT}/chkpnt${OPTIM_NUM_ITER}.pth
python stage3.py -s ${DATA_ROOT}/${DATA_FOLDER} --gen_root oc_generations/ --eval --exp_name ocmotion/${SUBJECT} --motion_offset_flag --iterations $NUM_ITER --white_background --test_iterations $((OPTIM_NUM_ITER+NUM_ITER)) --start_checkpoint $CKPT
python render.py -m output/ocmotion/${SUBJECT} --exp_name ocmotion/${SUBJECT} --motion_offset_flag --iteration $((OPTIM_NUM_ITER+NUM_ITER)) --white_background --skip_test 
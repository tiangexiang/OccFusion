DATA_ROOT=""
NUM_ITER=1800
OPTIM_NUM_ITER=1200


SUBJECT="0011_1"
# SUBJECT="0011_2"
# SUBJECT="0013"
# SUBJECT="0038"
# SUBJECT="0039"
# SUBJECT="0041"

python render.py -m output/ocmotion/${SUBJECT} --gen_root oc_generations/ --exp_name ocmotion/${SUBJECT} --motion_offset_flag --iteration $((OPTIM_NUM_ITER+NUM_ITER)) --white_background --skip_test 
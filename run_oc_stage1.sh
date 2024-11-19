NUM_ITER=1200
DATA_ROOT="data/"


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

python stage1.py -s ${DATA_ROOT}/${DATA_FOLDER} --gen_root oc_generations/ --eval --exp_name ocmotion/${SUBJECT} --motion_offset_flag --iterations $NUM_ITER --white_background --test_iterations $NUM_ITER
# render is NECCESARY for later stages
python render.py -m output/ocmotion/${SUBJECT} --motion_offset_flag --iteration $NUM_ITER --white_background --skip_test 
NUM_ITER=1200+1800


SUBJECT="0011_02_1_w2a"
# SUBJECT="0011_02_2_w2a"
# SUBJECT="0013_02_w2a"
# SUBJECT="0038_04_w2a"
# SUBJECT="0039_02_w2a"
# SUBJECT="0041_00_w2a"

python render.py -m output/ocmotion/${SUBJECT} --motion_offset_flag --iteration $NUM_ITER --white_background --skip_test

NUM_ITER=2800


SUBJECT="0011_1"
# SUBJECT="0011_2"
# SUBJECT="0041"
# SUBJECT="0039"
# SUBJECT="0038"
# SUBJECT="0013"

python render.py -m output/ocmotion/${SUBJECT} --motion_offset_flag --iteration $NUM_ITER --white_background --skip_test

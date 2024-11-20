DATA_ROOT="data/" # path to sequences
NUM_ITER=1200 # path to SAM-HQ checkpoint
STEP=35 # num of steps for stable diffusion inference


SUBJECT="0011_02_1_w2a"
# SUBJECT="0011_02_2_w2a"
# SUBJECT="0013_02_w2a"
# SUBJECT="0038_04_w2a"
# SUBJECT="0039_02_w2a"
# SUBJECT="0041_00_w2a"

######################################################
# The --overwrite flag refreshes existing generations.
######################################################

STAGE1_RENDERING=output/ocmotion/${SUBJECT}/train/${NUM_ITER}/renders/
python3 stage2_oc.py --subject $SUBJECT --data_root ${DATA_ROOT}/${SUBJECT} --stage1_render_path $STAGE1_RENDERING --num_inference_steps $STEP # --overwrite
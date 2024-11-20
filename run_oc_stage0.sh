DATA_ROOT="data/" # path to sequences
SAM="assets/sam_hq_vit_h.pth" # path to SAM-HQ checkpoint
STEP=15 # num of steps for stable diffusion inference


SUBJECT="0011_02_1_w2a"
# SUBJECT="0011_02_2_w2a"
# SUBJECT="0013_02_w2a"
# SUBJECT="0038_04_w2a"
# SUBJECT="0039_02_w2a"
# SUBJECT="0041_00_w2a"

######################################################
# The --overwrite flag refreshes existing generations.
######################################################

python3 stage0_oc.py --subject $SUBJECT --data_root ${DATA_ROOT}/${SUBJECT} --sam_checkpoint $SAM --num_inference_steps $STEP # --overwrite
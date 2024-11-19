DATA_ROOT=""
SAM=""
STEP=15


DATA_FOLDER="0011_02_1_w2a"
# DATA_FOLDER="0011_02_2_w2a"
# DATA_FOLDER="0013_02_w2a"
# DATA_FOLDER="0038_04_w2a"
# DATA_FOLDER="0039_02_w2a"
# DATA_FOLDER="0041_00_w2a"

python3 stage0_oc.py --subject $DATA_FOLDER --data_root ${DATA_ROOT}/${DATA_FOLDER} --sam_checkpoint $SAM --num_inference_steps $STEP # --overwrite
#TASK_NAME=SST-2
TASK_NAME=RTE
GLUE_DIR=data/glue_data
TC_DIR=tc_data

cd ..
python3 preprocess_tc.py --glue_dir $GLUE_DIR --save_dir $TC_DIR/glue_tc_preprocessed

cd scRNN
python3 train.py --task-name $TASK_NAME --preprocessed_glue_dir ../$TC_DIR/glue_tc_preprocessed --tc_dir ../$TC_DIR

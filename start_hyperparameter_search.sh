STUDY_NAME="search-1-0-all"
STORAGE="optuna_robot_detection"

echo $(which python)
echo "Running study: ${STUDY_NAME}"
CUDA_VISIBLE_DEVICES=0 nohup python -m src.train.hyperparameter_search ${STUDY_NAME} ${STORAGE} 0 > log_gpu0 2>&1 &
# Required if the database is created by first process
sleep 10
CUDA_VISIBLE_DEVICES=1 nohup python -m src.train.hyperparameter_search ${STUDY_NAME} ${STORAGE} 0 > log_gpu1 2>&1 &

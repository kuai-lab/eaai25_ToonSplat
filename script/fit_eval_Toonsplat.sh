dataset_mode="personal_data"
profile="people_1"
logbase=$1
cuda=$2

for seq in  "AladdinDance" "haku" "zelda" "3d_girl_walk" 
do
    CUDA_VISIBLE_DEVICES=$2 python solver_style.py --profile ./profiles/people/${profile}.yaml --dataset_mode $dataset_mode --seq $seq --logbase $logbase --fast --no_eval
    CUDA_VISIBLE_DEVICES=$2 python solver_style.py --profile ./profiles/people/${profile}.yaml --dataset_mode $dataset_mode --seq $seq --eval_only --log_dir logs/${logbase}/seq=${seq}_prof=${profile}_data=${dataset_mode}
    cd test_utils
    CUDA_VISIBLE_DEVICES=$2 python eval_metrics.py --rgb_gt_dir ../data/InstantAvatar_preprocessed_data_all/masked_image_test/$seq --rgb_pred_dir ../logs/${logbase}/seq=${seq}_prof=${profile}_data=${dataset_mode}/test_tto
    cd ..
done

 
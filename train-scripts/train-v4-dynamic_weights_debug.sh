cd /lp/pfld-multi
if [ ! -d ./output_dir ]
then
mkdir ./output_dir
fi
# expanded_dir=/lp/fastface.6.20/dataset/train-pfld-with-expandedbox/with-expandedbox

# train_file=${expanded_dir}/anno-train-onlyjd.txt
# test_file=${expanded_dir}/anno-test-jd10000.txt
# img_dir=${expanded_dir}/cropped-img

train_file=/lp/fastface.6.20/dataset/dataset-10.23/anno-no-aug-celeba-all-7attr-train.txt
# test_file=/lp/fastface.6.20/dataset/Test_data/Test_data1/anno-test-pfld-stlmk-having-gender.txt
test_file=/lp/fastface.6.20/dataset/dataset-10.23/anno-no-aug-celeba-all-7attr-test.txt
# anno-refine-for-eye.txt
# test_file=${train_file}
# train_file=${test_file}
# img_dir=/lp/fastface.6.20/dataset/dataset-10.23/cropped
# train_img_dir=/lp/fastface.6.20/dataset/dataset-10.23/cropped
# train_img_dir=/lp/fastface.6.20/dataset/dataset-10.23/celeba-all-cropped
train_img_dir=/lp/dataset-10.23/cropped_4_sets
test_img_dir=/lp/fastface.6.20/dataset/dataset-10.23/celeba-all-cropped
# test_img_dir=/lp/fastface.6.20/dataset/Test_data/Test_data1/cropped
# test_img_dir=${train_img_dir}
# train_img_dir=${test_img_dir}
# test_img_dir=${train_img_dir}

CUDA_VISIBLE_DEVICES=1
lr=1.25e-5
model_dir=./output_dir/model-v4_dyw_debug
# pretrained_model_dir=/lp/pfld/output_dir/model-only-jd-refine-eye-cls
pretrained_model_dir=${model_dir}
# pretrained_model_dir=./output_dir/model-v4-single-task-right-jd
tflog_dir=./output_dir/tflog-v4_dyw_debug
logs=./output_dir/train-v4_dyw_debug.log
tfport=8091
loss_type=2 #0 or 2

model_version=4

if [ ! -d ${model_dir} ]
then
mkdir ${model_dir}
fi

if [ ! -d ${tflog_dir} ]
then
mkdir ${tflog_dir}
fi

# nohup \
python3 -u train_model_v4_dynamic_weights_debug.py --learning_rate=${lr} \
                               --level L5 \
                               --image_size 112 \
                               --max_epoch 1600 \
                               --lr_epoch 100,200,300,400,500 \
                               --cuda-visibale-device ${CUDA_VISIBLE_DEVICES} \
                               --model_dir ${model_dir} \
                               --pretrained_model ${pretrained_model_dir} \
                               --file_list ${train_file} \
                               --img-dir ${train_img_dir} \
                               --log_dir ${tflog_dir} \
                               --test-img-dir ${test_img_dir} \
                               --test_list  ${test_file} \
                               --lmk-norm \
                               --loss-type ${loss_type} \
                               --model-version ${model_version} \
                               --mean-pts /lp/fastface.6.20/dataset/dataset-10.23/train-mean-pts.txt \
                               --train_file /lp/dataset-10.23/Test_data1908_mapped.csv \
                                            /lp/dataset-10.23/imdb_merged_united_cut_cut-invalid-male.train.csv \
                                            /lp/dataset-10.23/lfw_merged_united_cut.train.csv \
                                             /lp/dataset-10.23/celeba_merged_united_cut.train.csv \
                               --test_file /lp/dataset-10.23/Test_data1908_mapped.csv \
                                            /lp/dataset-10.23/imdb_merged_united_cut_cut-invalid-male.test-sample2000.csv \
                                            /lp/dataset-10.23/lfw_merged_united_cut.test.csv \
                                             /lp/dataset-10.23/celeba_merged_united_cut.test-sample2000.csv \
                               --batch_size   32 11 11 11 \
                               --train_data_type jd imdb lfw celeba \
                               --test_data_type jd imdb lfw celeba \
                               --weight_decay 1.25e-7 
                            #    >> ${logs} 2>&1 &
#  --no_addition 
# /lp/dataset-10.23/right-jd.csv \
                              #  /lp/dataset-10.23/imdb_merged_united_cut_cut-invalid-male.csv \
                              #                /lp/dataset-10.23/jd_merged_united_complete.csv \
                              #                /lp/dataset-10.23/lfw_merged_united_cut.csv \
                              #                /lp/dataset-10.23/celeba_merged_united_cut.csv \

                            #--train_file /lp/dataset-10.23/imdb_merged_united_cut_cut-invalid-male.csv /lp/dataset-10.23/jd_merged_united_complete.csv /lp/dataset-10.23/lfw_merged_united_cut.csv /lp/dataset-10.23/celeba_merged_united_cut.csv \
                            #    --train_file /lp/dataset-10.23/imdb_merged_united.csv /lp/dataset-10.23/celeba_merged_united.csv \

                            #    --to-train-prefix pfld_inference/fc2 
                            #    >> ${logs} 2>&1 &
                               #    --begin-multi \
#                               --test-img-dir /lp/fastface.6.20/dataset/output-test-img \
#                               --test_list /lp/fastface.6.20/dataset/test-anno-with-eulerangles-sample10000.txt \
# nohup tensorboard --logdir ${tflog_dir} --port ${tfport} >/dev/null 2>&1 &
# tail -f ${logs}


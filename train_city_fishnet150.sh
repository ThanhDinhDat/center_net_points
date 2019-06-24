cd src
# train
python main.py ctdet --arch fishnet_150 --exp_id coco_fishnet150 --batch_size 8 --master_batch 9 --lr 5e-4 --gpus 0 --num_workers 4 --num_epochs 2000 --lr_step 180,210 --debug 1

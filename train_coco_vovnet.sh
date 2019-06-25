cd src
# train
python main.py ctdet --exp_id coco_vovnet --arch vovnet_57 --batch_size 8 --master_batch 2 --lr 2.5e-4 --gpus 0 --num_workers 4 --num_epochs 500 --lr_step 180,210 --debug 1

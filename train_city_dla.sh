cd src
# train
python main.py motdet --dataset mot --exp_id city_dla_2x --batch_size 12 --master_batch 9 --lr 5e-4 --gpus 1 --num_workers 4 --num_epochs 2000 --lr_step 180,210 --resume


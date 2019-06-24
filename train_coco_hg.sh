cd src
# train
python main.py ctdet --exp_id coco_hg --arch hourglass --batch_size 2 --master_batch 2 --lr 2.5e-4 --load_model ../models/ExtremeNet_500000.pth --gpus 1


# new LWS and L2 :

#python train.py  --experts 2  --small_dataset True --model Deeplabv3_w_Better_Experts --lws True --L2penalty  True --log_wandb True  --name ace_2exp_new_lws_L2 --out_dir out/new_lws/
python train.py  --experts 3  --small_dataset True --model Deeplabv3_w_Better_Experts --lws True --L2penalty  True --log_wandb True  --name ace_3exp_new_lws_L2 --out_dir out/new_lws/

# Then we will work on frozen networks I guess : fine tune MLP or CNN, or use weight in the experts 3 we will need to look for the best learning rate.. maybe I can try to launch a sweep ?
python train.py  --experts 3   --log_wandb True  --out_dir out/cnn/ --CNN_aggregator True --pretrained_weights /home/valerie/Projects/Alps_LCC/out/cnn/ace_3exp_CNN_new_lws_L2/current_best.pt --finetune_classifier_only True --name finetune_cnn_v1
python train.py  --experts 3   --log_wandb True  --out_dir out/cnn/ --CNN_aggregator True --pretrained_weights /home/valerie/Projects/Alps_LCC/out/weight_decay/ace_3exp_wd_1e1/current_best.pt --finetune_classifier_only True --name finetune_cnn_v2





python train.py  --experts 3  --small_dataset True --model Deeplabv3_w_Better_Experts --lws True                   --log_wandb True  --name ace_3exp_new_lws --out_dir out/new_lws/





# Re run baseline 
python train.py --bs 64 --name deeplabv3plus_July3 --small_dataset True      --log_wandb True --out_dir out/baseline/

# Look for good weight decay
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True  --name ace_3exp_wd_005_lws_l2 --out_dir out/weight_decay/ --weight_decay 0.05 --lws True --L2penalty  True
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True  --name ace_3exp_wd_015_lws_l2 --out_dir out/weight_decay/ --weight_decay 0.15 --lws True --L2penalty  True
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True  --name ace_3exp_wd_020_lws_l2 --out_dir out/weight_decay/ --weight_decay 0.2  --lws True --L2penalty  True
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True  --name ace_3exp_wd_020        --out_dir out/weight_decay/ --weight_decay 0.2   


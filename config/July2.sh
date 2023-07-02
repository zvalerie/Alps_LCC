# new LWS and L2 :

python train.py  --experts 2  --small_dataset True --model Deeplabv3_w_Better_Experts --lws True --L2penalty  True --log_wandb True  --name ace_2exp_new_lws_L2 --out_dir out/new_lws/
python train.py  --experts 3  --small_dataset True --model Deeplabv3_w_Better_Experts --lws True --L2penalty  True --log_wandb True  --name ace_3exp_new_lws_L2 --out_dir out/new_lws/
python train.py  --experts 3  --small_dataset True --model Deeplabv3_w_Better_Experts --lws True                   --log_wandb True  --name ace_3exp_new_lws --out_dir out/new_lws/


# Re run baseline 
python train.py --bs 64 --name deeplabv3plus_July2 --small_dataset True      --log_wandb True --out_dir out/baseline/
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True  --name ace_3exp_wd_005 --out_dir out/weight_decay/ --weight_decay 0.05
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True  --name ace_3exp_wd_02 --out_dir out/weight_decay/ --weight_decay 0.2


# Then we will work on frozen networks I guess : fine tune MLP or CNN, or use weight in the experts
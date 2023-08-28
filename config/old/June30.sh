
# new lws : 
python train.py  --experts 2 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True     --L2penalty True      --lws True --name ace_2exp_L2_new_lws --out_dir out/lws/
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True     --L2penalty True      --lws True --name ace_3exp_L2_new_lws --out_dir out/lws/

# Catch up from june 28 :
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True     --L2penalty True  --separate_backprop True    --lws True --name ace_3exp_L2_new_lws_separate_bckprp --epoch 75


python train.py  --experts 2  --small_dataset True --CNN_aggregator True --model Deeplabv3_w_Better_Experts --lws True --L2penalty  True --log_wandb True  --name ace_2exp_CNN_new_lws_L2 --out_dir out/cnn/
python train.py  --experts 3  --small_dataset True --CNN_aggregator True --model Deeplabv3_w_Better_Experts --lws True --L2penalty  True --log_wandb True  --name ace_3exp_CNN_new_lws_L2 --out_dir out/cnn/


# Re-run baselines : 
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True  --name ace_3exp_30June --out_dir out/baseline/
python train.py  --experts 2 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True  --name ace_2exp_30June --out_dir out/baseline/

# Best model so far to re-run
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True     --L2penalty True  --separate_backprop True    --lws True --name ace_3exp_L2_new_lws_backprop

# Try different weight decay values : 
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True  --name ace_3exp_wd_0 --out_dir out/weight_decay/ --weight_decay 0.
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True  --name ace_3exp_wd_1e1 --out_dir out/weight_decay/ --weight_decay 1e-1
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True  --name ace_3exp_wd_1e3 --out_dir out/weight_decay/ --weight_decay 1e-3

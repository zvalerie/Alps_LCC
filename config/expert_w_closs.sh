#python train.py  --name ace_2exp_L2 --experts 2 --small_dataset True      --log_wandb True  --L2penalty True
#python train.py  --name ace_3exp_L2 --experts 3 --small_dataset True      --log_wandb True   --L2penalty  True

#python train.py  --name ace_2exp_better_L2 --experts 2 --small_dataset True      --log_wandb True --model Deeplabv3_w_Better_Experts    --L2penalty True
#python train.py  --name ace_3exp_better__L2 --experts 3 --small_dataset True      --log_wandb True --model Deeplabv3_w_Better_Experts    --L2penalty True


python train.py  --name ace_2exp_lws --experts 2 --small_dataset True      --log_wandb True  --lws True 
python train.py  --name ace_3exp_lws --experts 3 --small_dataset True      --log_wandb True   --lws True 

python train.py  --name ace_2exp_better_lws --experts 2 --small_dataset True      --log_wandb True --model Deeplabv3_w_Better_Experts    --lws True 
python train.py  --name ace_3exp_better_lws --experts 3 --small_dataset True      --log_wandb True --model Deeplabv3_w_Better_Experts    --lws True 



python train.py  --name ace_2exp_lws_L2 --experts 2 --small_dataset True      --log_wandb True  --lws True --L2penalty  True
python train.py  --name ace_3exp_lws_L2 --experts 3 --small_dataset True      --log_wandb True   --lws True --L2penalty  True

python train.py  --name ace_2exp_better_lws_L2 --experts 2 --small_dataset True      --log_wandb True --model Deeplabv3_w_Better_Experts    --lws True --L2penalty  True
python train.py  --name ace_3exp_better_lws_L2 --experts 3 --small_dataset True      --log_wandb True --model Deeplabv3_w_Better_Experts    --lws True --L2penalty  True
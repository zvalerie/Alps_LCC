#python train.py  --name ace_2exp_w_cl --experts 2 --small_dataset True      --log_wandb True  --L2penalty True
python train.py  --name ace_3exp_w_cl --experts 3 --small_dataset True      --log_wandb True   --L2penalty  True --debug True 

python train.py  --name ace_2exp_better_w_cl --experts 2 --small_dataset True      --log_wandb True --model Deeplabv3_w_Better_Experts    --L2penalty True --debug True 
python train.py  --name ace_3exp_better_w_cl --experts 3 --small_dataset True      --log_wandb True --model Deeplabv3_w_Better_Experts    --L2penalty True --debug True 


python train.py  --name ace_2exp_lws --experts 2 --small_dataset True      --log_wandb True  --lws True  --debug True 
python train.py  --name ace_3exp_lws --experts 3 --small_dataset True      --log_wandb True   --lws True  --debug True 

python train.py  --name ace_2exp_better_lws --experts 2 --small_dataset True      --log_wandb True --model Deeplabv3_w_Better_Experts    --lws True  --debug True 
python train.py  --name ace_3exp_better_lws --experts 3 --small_dataset True      --log_wandb True --model Deeplabv3_w_Better_Experts    --lws True --debug True 


python train.py --bs 64 --name ace_2exp --experts 2 --small_dataset True      --log_wandb True --debug True 
python train.py --bs 64 --name ace_3exp --experts 3 --small_dataset True      --log_wandb True --debug True 

python train.py --bs 64 --name ace_2exp_hlr --experts 2 --small_dataset True      --log_wandb True --debug True --lr 5e-4
python train.py --bs 64 --name ace_3exp_hlr --experts 3 --small_dataset True      --log_wandb True --debug True --lr 5e-4

python train.py --bs 64 --name ace_2exp_better --experts 2 --small_dataset True      --log_wandb True --model Deeplabv3_w_Better_Experts --debug True 
python train.py --bs 64 --name ace_3exp_better --experts 3 --small_dataset True      --log_wandb True --model Deeplabv3_w_Better_Experts --debug True 
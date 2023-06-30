# Follow up with the exp with L2 penalty and LWS
#python train.py  --name ace_2exp_lws_L2 --experts 2 --small_dataset True      --log_wandb True  --lws True --L2penalty  True   --out_dir
#python train.py  --name ace_3exp_lws_L2 --experts 3 --small_dataset True      --log_wandb True   --lws True --L2penalty  True  --out_dir


# experiments with CNN to merge : 

#python train.py  --experts 2  --small_dataset True --CNN_aggregator True --model Deeplabv3_w_Better_Experts  --log_wandb True  --name ace_2exp_CNN --out_dir out/cnn/
#python train.py  --experts 3  --small_dataset True --CNN_aggregator True --model Deeplabv3_w_Better_Experts  --log_wandb True  --name ace_3exp_CNN --out_dir out/cnn/


#python train.py  --experts 2  --small_dataset True --CNN_aggregator True --model Deeplabv3_w_Better_Experts --lws True --L2penalty  True --log_wandb True  --name ace_2exp_CNN_lws_L2 --out_dir out/cnn/
#python train.py  --experts 3  --small_dataset True --CNN_aggregator True --model Deeplabv3_w_Better_Experts --lws True --L2penalty  True --log_wandb True  --name ace_3exp_CNN_lws_L2 --out_dir out/cnn/

# Experiements with separate backpropagation for experts : 

#python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts  --log_wandb True    --L2penalty True   --separate_backprop True                --name ace_3exp_L2_separate_bckprp      --epoch 150
#python train.py  --experts 2 --small_dataset True   --model Deeplabv3_w_Better_Experts  --log_wandb True    --L2penalty True   --separate_backprop True                --name ace_2exp_L2_separate_bckprp_longer --epoch 75

#python train.py  --experts 2 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True                       --separate_backprop True                --name ace_2exp_separate_bckprp 
#python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True                       --separate_backprop True                --name ace_3exp_separate_bckprp 

#python train.py  --experts 2 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True     --L2penalty True  --separate_backprop True    --lws True --name ace_2exp_L2_lws_separate_bckprp --epoch 75
#python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True     --L2penalty True  --separate_backprop True    --lws True --name ace_3exp_L2_lws_separate_bckprp --epoch 75




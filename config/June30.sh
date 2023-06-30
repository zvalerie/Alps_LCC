

# Catch up from june 28 :
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True     --L2penalty True  --separate_backprop True    --lws True --name ace_3exp_L2_lws_separate_bckprp --epoch 75


python train.py  --experts 2  --small_dataset True --CNN_aggregator True --model Deeplabv3_w_Better_Experts --lws True --L2penalty  True --log_wandb True  --name ace_2exp_CNN_lws_L2 --out_dir out/cnn/
python train.py  --experts 3  --small_dataset True --CNN_aggregator True --model Deeplabv3_w_Better_Experts --lws True --L2penalty  True --log_wandb True  --name ace_3exp_CNN_lws_L2 --out_dir out/cnn/



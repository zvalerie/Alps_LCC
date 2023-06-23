python train.py --bs 64 --name deeplabv3plus_new_scheduler --small_dataset True      --log_wandb True 


#python train.py --bs 64 --name deeplabv3plus_base_all      --log_wandb True --epoch 15   --step_size 5
python train.py --bs 64 --name unet_base_all  --model Unet --log_wandb True --epoch 15   --step_size 5

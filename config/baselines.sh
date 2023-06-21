python train.py --bs 64 --name deeplabv3plus_base  --log_wandb True
python train.py --bs 64 --name unet_base  --model Unet    --log_wandb True

python train.py --bs 64 --name deeplabv3plus_base_all  --small_dataset False   --log_wandb True
python train.py --bs 64 --name unet_base_all  --model Unet  --small_dataset False   --log_wandb True
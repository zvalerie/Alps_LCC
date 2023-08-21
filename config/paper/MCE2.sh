
python train.py --bs 64 --name mce2_base_opt --experts 2   --log_wandb True  --model MCE --out_dir out/mce2/  
python train.py --bs 64 --name mce2_lws_opt --experts 2   --log_wandb True  --model MCE --out_dir out/mce2/  --lws True
python train.py --bs 64 --name mce2_l2_opt --experts 2    --log_wandb True  --model MCE --out_dir out/mce2/  --L2penalty True 
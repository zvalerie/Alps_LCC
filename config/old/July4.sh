
# look for best learning rate for finetuning cnn on top of frozen network
python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_cnn/ --CNN_aggregator True --finetune_classifier_only True --name ft_cnn_lr_1e_1 --lr 0.1             --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 
python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_cnn/ --CNN_aggregator True --finetune_classifier_only True --name ft_cnn_lr_1e_1 --lr 1e-2             --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 
python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_cnn/ --CNN_aggregator True --finetune_classifier_only True --name ft_cnn_lr_1e_1 --lr 1e-3             --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 
python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_cnn/ --CNN_aggregator True --finetune_classifier_only True --name ft_cnn_lr_1e_1 --lr 1e-4             --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 
python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_cnn/ --CNN_aggregator True --finetune_classifier_only True --name ft_cnn_lr_1e_1 --lr 1e-5            --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 
python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_cnn/ --CNN_aggregator True --finetune_classifier_only True --name ft_cnn_lr_1e_1 --lr 1e-6             --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 




# look for best learning rate for finetuning cnn on top of frozen network
python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_mlp/  --MLP_aggregator True --finetune_classifier_only True --name ft_mlp_lr_1e_1 --lr 0.1             --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 
python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_mlp/  --MLP_aggregator True --finetune_classifier_only True --name ft_mlp_lr_1e_1 --lr 1e-2             --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 
python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_mlp/  --MLP_aggregator True --finetune_classifier_only True --name ft_mlp_lr_1e_1 --lr 1e-3             --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 
python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_mlp/  --MLP_aggregator True --finetune_classifier_only True --name ft_mlp_lr_1e_1 --lr 1e-4             --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 
python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_mlp/  --MLP_aggregator True --finetune_classifier_only True --name ft_mlp_lr_1e_1 --lr 1e-5            --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 
python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_mlp/  --MLP_aggregator True --finetune_classifier_only True --name ft_mlp_lr_1e_1 --lr 1e-6             --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 


# missing run : 

python train.py  --experts 3  --small_dataset True --model Deeplabv3_w_Better_Experts --lws True                   --log_wandb True  --name ace_3exp_new_lws --out_dir out/new_lws/





# Re run baseline 
python train.py --bs 64 --name deeplabv3plus_July3 --small_dataset True      --log_wandb True --out_dir out/baseline/

# Look for good weight decay
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True  --name ace_3exp_wd_005_lws_l2 --out_dir out/weight_decay/ --weight_decay 0.05 --lws True --L2penalty  True
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True  --name ace_3exp_wd_015_lws_l2 --out_dir out/weight_decay/ --weight_decay 0.15 --lws True --L2penalty  True
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True  --name ace_3exp_wd_020_lws_l2 --out_dir out/weight_decay/ --weight_decay 0.2  --lws True --L2penalty  True
python train.py  --experts 3 --small_dataset True   --model Deeplabv3_w_Better_Experts   --log_wandb True  --name ace_3exp_wd_020        --out_dir out/weight_decay/ --weight_decay 0.2   


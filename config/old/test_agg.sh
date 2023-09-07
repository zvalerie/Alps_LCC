######### Balanced CNN merge CBL 

python train.py --experts 3 --aggregation MLP_select   --epoch 25   --lr 0.01     --name MLP_select_lr1e_2_sgd   --log_wandb True   --weight_decay 0.2    --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt 

python train.py --experts 3 --aggregation MLP_select   --epoch 25   --lr 0.001    --name MLP_select_lr1e_3_sgd   --log_wandb True   --weight_decay 0.2    --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt 

python train.py --experts 3 --aggregation MLP_select   --epoch 25   --lr 0.0001   --name MLP_select_lr1e_4_sgd   --log_wandb True   --weight_decay 0.2    --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt 

python train.py --experts 3 --aggregation MLP_select   --epoch 25   --lr 0.00001  --name MLP_select_lr1e_5_sgd   --log_wandb True   --weight_decay 0.2    --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt




######### Balanced MLP merge CBL

#python train.py --experts 3 --aggregation MLP_merge   --epoch 50 --reweighted_aggregation CBL   --lr 0.00001   --name MLP_merge_CBL_lr_1e-5   --log_wandb True      --weight_decay 0.2    --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt 

#python train.py --experts 3 --aggregation MLP_merge   --epoch 50 --reweighted_aggregation CBL   --lr 0.00005   --name MLP_merge_CBL_lr_5e-5   --log_wandb True      --weight_decay 0.2    --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt 

#python train.py --experts 3 --aggregation MLP_merge   --epoch 50 --reweighted_aggregation CBL   --lr 0.000001  --name MLP_merge_CBL_lr_1e-6   --log_wandb True      --weight_decay 0.2    --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt 


#### inverse_frequency 

#python train.py --experts 3 --aggregation MLP_merge   --epoch 50 --reweighted_aggregation inverse_frequency   --lr 0.00001   --name MLP_merge_invf_lr_1e-5   --log_wandb True     --weight_decay 0.2    --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt 

#python train.py --experts 3 --aggregation MLP_merge   --epoch 50 --reweighted_aggregation inverse_frequency   --lr 0.00005   --name MLP_merge_invf_lr_5e-5   --log_wandb True     --weight_decay 0.2     --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt 

#python train.py --experts 3 --aggregation MLP_merge   --epoch 50 --reweighted_aggregation inverse_frequency   --lr 0.000001  --name MLP_merge_invf_lr_1e-6   --log_wandb True     --weight_decay 0.2     --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt 


######### Balanced CNN merge CBL 

python train.py --experts 3 --aggregation CNN_merge   --epoch 50 --reweighted_aggregation CBL   --lr 0.00001  --name CNN_merge_CBL_lr_1e-5   --log_wandb True   --weight_decay 0.2    --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt 



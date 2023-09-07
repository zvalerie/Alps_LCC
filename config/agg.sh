### test mlp merge to reach former exp levels 


#python train.py --experts 3 --aggregation MLP_merge   --epoch 50 --reweighted_aggregation CBL    --lr 0.0005   --name MLP_merge_CBL   --log_wandb True     --weight_decay 0.2    --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt 

#python train.py --experts 3 --aggregation MLP_merge   --epoch 50 --reweighted_aggregation None   --lr 0.00005   --name MLP_merge_no_w   --log_wandb True     --weight_decay 0.2    --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt 

python train.py --experts 3 --aggregation CNN_merge   --epoch 50 --reweighted_aggregation CBL    --lr 0.0005   --name CNN_merge_CBL   --log_wandb True     --weight_decay 0.2    --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt 

python train.py --experts 3 --aggregation CNN_merge   --epoch 50 --reweighted_aggregation None   --lr 0.00005   --name CNN_merge_no_w   --log_wandb True     --weight_decay 0.2    --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt 


python train.py --experts 3 --aggregation CNN_merge   --epoch 50 --reweighted_aggregation CBL    --lr 0.000005   --name CNN_merge_CBL_1e6   --log_wandb True     --weight_decay 0.2    --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt 

python train.py --experts 3 --aggregation CNN_merge   --epoch 50 --reweighted_aggregation None   --lr 0.0000005   --name CNN_merge_no_w_1e6   --log_wandb True     --weight_decay 0.2    --lws True   --finetune_classifier_only True  --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt 



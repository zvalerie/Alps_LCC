
# Run each in different terminal
### MLP MOE with lr =1e-2 and 5e-3 on large dataset


# terminal   mlp-moe-2
# python train.py --experts 3 --reweighted_aggregation CBL --aggregation MLP_moe --epoch 50 --lr 0.005     --name  MLP_moe_lr_5e-3_CBL_large     --finetune_classifier_only True --pretrained_weights  '/home/valerie/project/Alps_LCC/out/MCE3_moptim/final.pt'  --log_wandb True --large_dataset True  --weight_decay 0.2

# treminal train : 
# python train.py --experts 3 --reweighted_aggregation CBL --aggregation MLP_moe --epoch 50 --lr 0.01      --name  MLP_moe_lr_1e-2_CBL_large     --finetune_classifier_only True --pretrained_weights  '/home/valerie/project/Alps_LCC/out/MCE3_moptim/final.pt'  --log_wandb True  --large_dataset True --weight_decay 0.2



###CNN merge on large dataset :with lr : 0.0005
 ### terinal cnn_merge
python train.py --experts 3 --aggregation CNN_merge --epoch 50 --reweighted_aggregation CBL --lr 0.00005 --name CNN_merge_CBL_large --log_wandb True --weight_decay 0.2 --lws True --finetune_classifier_only True --pretrained_weights /home/valerie/Projects/Alps_LCC/out/august/MCE3_moptim/current_best.pt  --large_dataset True




### full network training 


python train.py --experts 3 --epoch 50 L --lr 0.000005 --name MCE3_macc --log_wandb True --weight_decay 0.2 --lws True  --large_dataset True  --L2penalty True

python train.py --experts 3 --reweighted_aggregation CBL --aggregation MLP_moe --epoch 50 --lr 0.005     --name  full_MLP_moe_lr_5e-3_CBL_large --log_wandb True --large_dataset True  --weight_decay 0.2 --lws True --L2penalty True

python train.py --experts 3 --aggregation CNN_merge --epoch 50 --reweighted_aggregation CBL --lr 0.0005 --name full_CNN_merge_CBL_large --log_wandb True --weight_decay 0.2 --lws True  --large_dataset True  --L2penalty True


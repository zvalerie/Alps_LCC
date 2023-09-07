
python train.py --experts 3 --aggregation MLP_moe --epoch 50 --lr 0.01      --name  MLP_moe_lr_1e-2     --finetune_classifier_only True --pretrained_weights  '/home/valerie/project/Alps_LCC/out/MCE3_moptim/final.pt'  --log_wandb True
python train.py --experts 3 --aggregation MLP_moe --epoch 50 --lr 0.05      --name  MLP_moe_lr_5e-2     --finetune_classifier_only True --pretrained_weights  '/home/valerie/project/Alps_LCC/out/MCE3_moptim/final.pt'  --log_wandb True
python train.py --experts 3 --aggregation MLP_moe --epoch 50 --lr 0.005     --name  MLP_moe_lr_5e-3     --finetune_classifier_only True --pretrained_weights  '/home/valerie/project/Alps_LCC/out/MCE3_moptim/final.pt'  --log_wandb True
python train.py --experts 3 --aggregation MLP_moe --epoch 50 --lr 0.001     --name  MLP_moe_lr_1e-3     --finetune_classifier_only True --pretrained_weights  '/home/valerie/project/Alps_LCC/out/MCE3_moptim/final.pt'  --log_wandb True

python train.py --experts 3 --aggregation MLP_moe --epoch 50 --lr 0.005     --name  MLP_moe_lr_5e-3_large     --finetune_classifier_only True --pretrained_weights  '/home/valerie/project/Alps_LCC/out/MCE3_moptim/final.pt'  --log_wandb True --large_dataset True

python train.py --experts 3 --reweighted_aggregation CBL --aggregation MLP_moe --epoch 50 --lr 0.001     --name  MLP_moe_lr_1e-3_CBL     --finetune_classifier_only True --pretrained_weights  '/home/valerie/project/Alps_LCC/out/MCE3_moptim/final.pt'  --log_wandb True
python train.py --experts 3 --reweighted_aggregation CBL --aggregation MLP_moe --epoch 50 --lr 0.01      --name  MLP_moe_lr_1e-2_CBL     --finetune_classifier_only True --pretrained_weights  '/home/valerie/project/Alps_LCC/out/MCE3_moptim/final.pt'  --log_wandb True


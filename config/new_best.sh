## BEST :
# train.py --name MCE3_lcom_small --experts 3 --log_wandb True --L2penalty True 

## run baseline with 2 experts with new best
#python train.py --name MCE-2_lcom_small --experts 2 --log_wandb True --L2penalty True --large_dataset True --catchup_training True

python train.py --name MCE-3_lcom_longer --experts 3 --log_wandb True --L2penalty True --large_dataset True --catchup_training True --epoch 30 --aggregation mean


## run test max logits 
#python train.py --name mce_lcom_ws13 --expert 3 --log_wandb True --L2penalty True --test_only True  --out_dir out/best/  --pretrained_weights  /home/valerie/Projects/Alps_LCC/out/best/mce_lcom_ws13/current_best.pt
#python train.py --name mce_lcom_ws13 --expert 3 --log_wandb True  --aggregation max_pool --L2penalty True --test_only True  --out_dir out/best/  --pretrained_weights  /home/valerie/Projects/Alps_LCC/out/best/mce_lcom_ws13/current_best.pt


# # run test with zero non target

# python train.py --name mce_lcom_ws13 --expert 3 --log_wandb True --zero_nontarget_expert True --aggregation mean --L2penalty True --test_only True  --out_dir out/best/  --pretrained_weights  /home/valerie/Projects/Alps_LCC/out/best/mce_lcom_ws13/current_best.pt





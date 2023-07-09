
# look for best learning rate for finetuning cnn on top of frozen network
#python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_cnn/ --CNN_aggregator True --finetune_classifier_only True --name ft_cnn_wd_02 --lr 1e-5 --weight_decay 0.2         --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 


#python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_cnn/ --CNN_aggregator True --finetune_classifier_only True --name ft_cnn_wd_03 --lr 1e-5 --weight_decay 0.3         --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 


# now we have weights  :
#python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_cnn/ --CNN_aggregator True --finetune_classifier_only True --name ft_cnn_weights_wd_02 --lr 1e-5 --weight_decay 0.2         --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 

#python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_cnn/ --CNN_aggregator True --finetune_classifier_only True --name ft_cnn_weights --lr 1e-5 --weight_decay 0.         --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 

#python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_cnn/ --CNN_aggregator True --finetune_classifier_only True --name ft_cnn_weights_hlr --lr 1e-3 --weight_decay 0.         --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 

# test with drop out
python train.py  --experts 3   --log_wandb True  --out_dir out/aggregate_cnn/ --CNN_aggregator True --finetune_classifier_only True --name ft_cnn_weights_dropout --lr 1e-5 --weight_decay 0.2         --pretrained_weights /home/valerie/Projects/Alps_LCC/out/new_lws/ace_3exp_new_lws_L2/current_best.pt 


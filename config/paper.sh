###################################

## CEL
#python train.py --name CEL --experts 0    --log_wandb True  --large_dataset True

## WCEL
#python train.py --name WCEL --experts 0    --log_wandb True  --large_dataset True     --loss inverse_freq_weights 

## SL
#python train.py --name SL --experts 0    --log_wandb True  --large_dataset True     --loss seesaw
#TODO
## MCE 2
#python train.py --name MCE2 --experts 2  --log_wandb True  --lws True --L2penalty True --weight_decay 0.2   --large_dataset True 

## MCE 3 
#python train.py --name MCE3 --experts 3  --log_wandb True  --lws True --L2penalty True --weight_decay 0.2  --weight_decay 0.2  --large_dataset True 

#############################

## MCE base
#python train.py --name CEL --experts 3  --log_wandb True --large_dataset True 

## MCE +LWS
#python train.py --name CEL --experts 3  --log_wandb True --lws True     --large_dataset True 

## MCE +LCom
#python train.py --name CEL --experts 3  --log_wandb True --L2penalty True  --large_dataset True 


### MCE  + smaller batch size : 

#python train.py --name CEL_small_bs --experts 3  --log_wandb True --large_dataset True  --bs 16

## MCE + Lcon 


## MCE no WDT 
python train.py --name mce_no_wd --experts 3  --log_wandb True  --lws True --L2penalty True  --weight_decay 0.  --large_dataset True 

### Non adaptive LR
python train.py --name mce_not_adaptive_kr --experts 3  --log_wandb True  --lws True --L2penalty True   --not_adaptive_lr True  --large_dataset True 

## MCE + smaller batch size 
#python train.py --name mce_small_bs --experts 3  --log_wandb True --bs 8



############################# Aggregations methods


# MCE + max_pool output
#python train.py --name MCE3_max_pool --experts 3  --log_wandb True  --lws True --L2penalty True --weight_decay 0.2    --large_dataset True  --aggregation max_pool

# MCE + merge MLP
#python train.py  --log_wandb True --experts 3  --lws True --L2penalty True --weight_decay 0.2 --aggregation CNN_merge --name CNN_merge_sm

#MCE + merge CNN
#python train.py  --log_wandb True --experts 3  --lws True --L2penalty True --weight_decay 0.2 --aggregation MLP_merge --name MLP_merge_sm

# MCE + vote MLP
#python train.py  --log_wandb True --experts 3  --lws True --L2penalty True --weight_decay 0.2 --aggregation CNN_select --name CNN_select_sm

# MCE + vote CNN
#python train.py  --log_wandb True --experts 3  --lws True --L2penalty True --weight_decay 0.2 --aggregation MLP_select --name MLP_select_sm

## MCE 3 
#python train.py --name MCE3_moptim --experts 3  --log_wandb True  --lws True --L2penalty True --weight_decay 0.2  --weight_decay 0.2  --large_dataset True 

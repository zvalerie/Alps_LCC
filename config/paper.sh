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
python train.py --name MCE3 --experts 3  --log_wandb True  --lws True --L2penalty True --weight_decay 0.2  --weight_decay 0.2  --large_dataset True 

#############################

## MCE base
python train.py --name CEL --experts 3  --log_wandb True --large_dataset True 

## MCE +LWS
python train.py --name CEL --experts 3  --log_wandb True --lws True     --large_dataset True 

## MCE +LCom
python train.py --name CEL --experts 3  --log_wandb True --L2penalty True  --large_dataset True 


## MCE  + smaller batch size : 

python train.py --name CEL_small_bs --experts 3  --log_wandb True --large_dataset True  --bs 16

## MCE + Lcon 


## MCE no WDT 
python train.py --name mce_no_wd --experts 3  --log_wandb True  --lws True --L2penalty True  --weight_decay 0.  --large_dataset True 

## MCE + smaller batch size 
python train.py --name mce_small_bs --experts 3  --log_wandb True --bs 8



############################# Aggregations methods

# MCE + merge MLP
#python train.py --finetune_classifier_only  --log_wandb True 

#MCE + merge CNN
#python train.py --finetune_classifier_only  --log_wandb True 

# MCE + vote MLP
#python train.py --finetune_classifier_only  --log_wandb True 

# MCE + vote CNN
#python train.py --finetune_classifier_only   --log_wandb True


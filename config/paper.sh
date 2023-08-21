###################################

## CEL
#python train.py --name CEL --experts 0    --log_wandb True  --large_dataset True

## WCEL
#python train.py --name WCEL --experts 0    --log_wandb True  --large_dataset True     --inv_freq_weights 
#TODO
## SL
#python train.py --name SL --experts 0    --log_wandb True  --large_dataset True     --loss seesaw
#TODO
## MCE 2
python train.py --name MCE2 --experts 2  --lws True --L2penalty True --weight_decay 0.2   --large_dataset True

## MCE 3 
python train.py --name MCE3 --experts 3  --lws True --L2penalty True --weight_decay 0.2  --weight_decay 0.2  --large_dataset True 

#############################

## MCE base
python train.py --name CEL --experts 3 --large_dataset True 

## MCE +LWS
python train.py --name CEL --experts 3 --lws True     --large_dataset True 

## MCE +LCom
python train.py --name CEL --experts 3 --L2penalty True  --large_dataset True 

## MCE + Lcon 
??

## MCE + WDT 
python train.py --name mce_no_wd --experts 3  --lws True --L2penalty True  --weight_decay 0.  --large_dataset True 


############################# Aggregations methods

# MCE + merge MLP
#python train.py --finetune_classifier_only 

#MCE + merge CNN
#python train.py --finetune_classifier_only 

# MCE + vote MLP
#python train.py --finetune_classifier_only 

# MCE + vote CNN
#python train.py --finetune_classifier_only 


## CEL
python train.py --name CEL_ws13 --experts 0    --log_wandb True  --large_dataset True  


## SL
python train.py --name SL_ws13 --experts 0    --log_wandb True  --large_dataset True     --loss seesaw   

## MCE 2
python train.py --name MCE2_ws13 --experts 2  --log_wandb True  --lws True --L2penalty True --weight_decay 0.2   --large_dataset True    

## MCE 3 
#python train.py --name MCE3_ws13 --experts 3  --log_wandb True  --lws True --L2penalty True --weight_decay 0.2  --weight_decay 0.2  --large_dataset True   


## MCE base
python train.py --name MCE_base_ws13 --experts 3  --log_wandb True --large_dataset True   

## MCE +LWS
python train.py --name MCE_lws_ws13 --experts 3  --log_wandb True --lws True     --large_dataset True  

## MCE +LCom
python train.py --name MCE_lcom_ws13 --experts 3  --log_wandb True --L2penalty True  --large_dataset True   


## MCE  + smaller batch size : 

python train.py --name CEL_bs8_ws13 --experts 3  --log_wandb True --large_dataset True  --bs 16  
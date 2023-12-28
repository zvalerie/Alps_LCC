#simple CE 
python train.py --name CEL_tiny  --experts 0  --loss celoss    

# weighted CE
python train.py --name CEL_tiny --experts 0  --large_dataset False     --loss inverse_freq_weights 

## SL
python train.py --name CEL_tiny   --experts 0  --large_dataset False     --loss seesaw

## MCE 2
python train.py --name CEL_tiny --experts 2  --large_dataset False --lws True --L2penalty True --weight_decay 0.2   

## MCE 3 
python train.py --name CEL_tiny --experts 3  --large_dataset False --lws True --L2penalty True --weight_decay 0.2   



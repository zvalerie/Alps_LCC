# Just to debug and observe if everything works : 


#simple CE 
python train.py --name FLAIR_CEL_dev  --experts 0  --loss celoss   --bs 16   --log_wand True

# weighted CE
python train.py --name FLAIR_WCEL_dev   --experts 0    --loss inverse_freq_weights --debug True   --log_wand True

# Class balanced loss : 
#python train.py --name FLAIR_CBL_dev   --experts 0    --loss cbloss --debug True   --log_wand True

## SL
python train.py --name FLAIR_SL_dev  --experts 0  --loss seesaw   --log_wand True

## MCE 2
#python train.py --name FLAIR_MCE2_dev  --experts 2   --lws False --L2penalty True --bs 16     --log_wand True

## MCE 3 
python train.py --name FLAIR_MCE3_dev --experts 3   --lws False --L2penalty True  --bs 16     --log_wand True 


#####################################################
# Should produce the results for the papers on FLAIR: 
#######################################################

#simple CE 
#python train.py --name CEL_base  --experts 0  --loss celoss    --large_dataset True  --log_wand True  --bs 32

# weighted CE
#python train.py --name WEL_base   --experts 0    --loss inverse_freq_weights --large_dataset True  --log_wand True   --bs 32

# Class balanced loss : 
#python train.py --name CBL_base   --experts 0    --loss cbloss --large_dataset True  --log_wand True

## SL
#python train.py --name SL_base  --experts 0  --loss seesaw --large_dataset True  --log_wand True

## MCE 2
#python train.py --name MCE2_base  --experts 2   --lws False --L2penalty True    --large_dataset True  --log_wand True

## MCE 3 
#python train.py --name MCE3_base --experts 3   --lws False --L2penalty True    --large_dataset True  --log_wand True 

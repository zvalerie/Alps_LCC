# Just to debug and observe if everything works : python train.py --ds TLM --backbone unet


#simple CE 
#python train.py --ds TLM --backbone unet  --name TLM_unet_CEL_dev    --experts 0   --loss celoss       --log_wand True 

# weighted CE
#python train.py --ds TLM --backbone unet --name TLM_unet_WCEL_dev    --experts 0   --loss inverse_freq_weights    --log_wand True --lr 0.0001

# Class balanced loss : 
#python train.py --ds TLM --backbone unet --name TLM_unet_CBL_dev     --experts 0   --loss cbloss    --log_wand True

## SL
#python train.py --ds TLM --backbone unet --name TLM_unet_SL_dev      --experts 0   --loss seesaw    --log_wand True

## MCE 2
#python train.py --ds TLM --backbone unet --name TLM_unet_MCE2_dev_v2    --experts 2   --lws False --L2penalty True       --log_wand True 

## MCE 3 
#python train.py --ds TLM --backbone unet --name TLM_unet_MCE3_dev_v2    --experts 3   --lws False --L2penalty True       --log_wand True  --bs 32


#####################################################
# Should produce the results for the papers on FLAIR: 
#######################################################

#simple CE 
#python train.py --ds TLM --backbone unet --name TLM_unet_CEL_basev2    --experts 0   --loss celoss    --large_dataset True  --log_wand True

# weighted CE
#python train.py --ds TLM --backbone unet --name TLM_unet_WEL_base    --experts 0   --loss inverse_freq_weights  --large_dataset True  --log_wand True  --lr 0.00001

# Class balanced loss : 
python train.py --ds TLM --backbone unet --name TLM_unet_CBL_base    --experts 0   --loss cbloss --large_dataset True  --log_wand True --lr 0.00001

## SL
python train.py --ds TLM --backbone unet --name TLM_unet_SL_base     --experts 0   --loss seesaw --large_dataset True  --log_wand True --lr 0.00001

## MCE 2
#python train.py --ds TLM --backbone unet --name TLM_unet_MCE2_base   --experts 2   --lws False --L2penalty True    --large_dataset True  --log_wand True

## MCE 3 
#python train.py --ds TLM --backbone unet --name TLM_unet_MCE3_base   --experts 3   --lws False --L2penalty True    --large_dataset True  --log_wand True


#sudo systemctl restart display-manager
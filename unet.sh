# Just to debug and observe if everything works : python train.py --ds TLM --backbone unet


#simple CE 
python train.py --ds TLM --backbone unet  --name TLM_unet_CEL_dev    --experts 0   --loss celoss    --debug True   --log_wand True 

# weighted CE
python train.py --ds TLM --backbone unet --name TLM_unet_WCEL_dev    --experts 0   --loss inverse_freq_weights --debug True   --log_wand True

# Class balanced loss : 
python train.py --ds TLM --backbone unet --name TLM_unet_CBL_dev     --experts 0   --loss cbloss --debug True   --log_wand True

## SL
python train.py --ds TLM --backbone unet --name TLM_unet_SL_dev      --experts 0   --loss seesaw --debug True   --log_wand True

## MCE 2
python train.py --ds TLM --backbone unet --name TLM_unet_MCE2_dev    --experts 2   --lws True --L2penalty True --weight_decay 0.2    --debug True   --log_wand True

## MCE 3 
python train.py --ds TLM --backbone unet --name TLM_unet_MCE3_dev    --experts 3   --lws True --L2penalty True --weight_decay 0.2   --debug True   --log_wand True


#####################################################
# Should produce the results for the papers on FLAIR: 
#######################################################

#simple CE 
#python train.py --ds TLM --backbone unet --name TLM_unet_CEL_base    --experts 0   --loss celoss    --large_dataset True  --log_wand True

# weighted CE
#python train.py --ds TLM --backbone unet --name TLM_unet_WEL_base    --experts 0   --loss inverse_freq_weights --large_dataset True  --log_wand True

# Class balanced loss : 
#python train.py --ds TLM --backbone unet --name TLM_unet_CBL_base    --experts 0   --loss cbloss --large_dataset True  --log_wand True

## SL
#python train.py --ds TLM --backbone unet --name TLM_unet_SL_base     --experts 0   --loss seesaw --large_dataset True  --log_wand True

## MCE 2
#python train.py --ds TLM --backbone unet --name TLM_unet_MCE2_base   --experts 2   --lws True --L2penalty True --weight_decay 0.2    --large_dataset True  --log_wand True

## MCE 3 
#python train.py --ds TLM --backbone unet --name TLM_unet_MCE3_base   --experts 3   --lws True --L2penalty True --weight_decay 0.2   --large_dataset True  --log_wand True

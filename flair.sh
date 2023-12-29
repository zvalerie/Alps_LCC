#simple CE 
python train.py --name CEL_dev  --experts 0  --loss celoss    --debug True   --log_wand True

# weighted CE
python train.py --name WEL_dev   --experts 0    --loss inverse_freq_weights --debug True   --log_wand True

## SL
python train.py --name SL_dev  --experts 0  --loss seesaw --debug True   --log_wand True

## MCE 2
python train.py --name MCE2_dev  --experts 2   --lws True --L2penalty True --weight_decay 0.2    --debug True   --log_wand True

## MCE 3 
python train.py --name MCE3_dev --experts 3   --lws True --L2penalty True --weight_decay 0.2   --debug True   --log_wand True



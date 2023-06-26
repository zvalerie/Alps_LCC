python train.py --bs 64 --name deeplabv3plus_new_scheduler --small_dataset True                                             --test_only True  --out_dir 'out/baseline' 
python train.py --bs 64 --name deeplabv3plus_base_all                                                                       --test_only True  --out_dir 'out/baseline' 
python train.py --bs 64 --name unet_base  --model Unet                                                                  --test_only True  --out_dir 'out/baseline' 


python train.py --bs 64 --name ace_2exp --experts 2 --small_dataset True                                                    --test_only True  --out_dir 'out/experts' 
python train.py --bs 64 --name ace_3exp --experts 3 --small_dataset True                                                    --test_only True  --out_dir 'out/experts' 

python train.py --bs 64 --name ace_2exp_hlr --experts 2 --small_dataset True                                                --test_only True  --out_dir 'out/experts' 
python train.py --bs 64 --name ace_3exp_hlr --experts 3 --small_dataset True                                                --test_only True  --out_dir 'out/experts' 

python train.py --bs 64 --name ace_2exp_better --experts 2 --small_dataset True  --model Deeplabv3_w_Better_Experts         --test_only True  --out_dir 'out/experts' 
python train.py --bs 64 --name ace_3exp_better --experts 3 --small_dataset True  --model Deeplabv3_w_Better_Experts         --test_only True  --out_dir 'out/experts' 
clear
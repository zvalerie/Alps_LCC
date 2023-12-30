import torch
import wandb 
from sys import path as sys_path
sys_path.append('/home/valerie/Projects/contrastive-segmentation/contrastive-lc/') 

def resume_wandb_test(project_name,run_id,metrics ):




    wandb.finish()
    try : 
        wandb.init(project = project_name, 
                    id=run_id , 
                    entity = "zvalerie",
                    resume="must")
        
        print('Resume wandb run from id ',run_id )
    except : 
        FileNotFoundError 
        wandb.finish()

    wandb.log(metrics,)    

    wandb.finish()
    print('new values saved')

if __name__ =="__main__":
    project_name  = "ACE_revision_TLM"
    metrics = {
                    'test_miou' :0.128, 
                    'test_macc': 0.202 , 
                    'test_oacc': 0.484 ,
                    'frequent_cls_acc':0.434,
                    'common_cls_acc': 0.027,
                    'rare_cls_acc': 0.0,
                    
            }

    resume_wandb_test(project_name,
                      run_id='ppccoilk',
                      metrics=metrics
                      )


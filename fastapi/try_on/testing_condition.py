import os

if __name__ == '__main__':


    print('5')
 # Triaing codition
    print("\nRun HR-VITON to generate final image\n")

    terminnal_command = "python ./HR-VITON-main/train_condition.py --cuda True --name test5 --gpu_ids 0 --tocg_checkpoint ./HR-VITON-main/eval_models/weights/v0.1/tocg_final.pth --dataroot ./data/ --test_dataroot ./data/ --Ddownx2 --Ddropout --lasttvonly --interflowloss --occlusion"
    os.system(terminnal_command)
    print('6')

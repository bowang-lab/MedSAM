"""
The code was adapted from the MICCAI FLARE Challenge
https://github.com/JunMa11/FLARE

The testing images will be evaluated one by one.

Folder structure:
CVPR24_time_eval.py
- team_docker
    - teamname.tar.gz # submitted docker containers from participants
- test_demo
    - imgs
        - case1.npz  # testing image
        - case2.npz  
        - ...   
- demo_seg  # segmentation results
    - case1.npz  # segmentation file name is the same as the testing image name
    - case2.npz  
    - ...
"""

import os
join = os.path.join
import shutil
import time
import torch
import argparse
from collections import OrderedDict
import pandas as pd

parser = argparse.ArgumentParser('Segmentation efficiency eavluation for docker containers', add_help=False)
parser.add_argument('-i', '--test_img_path', default='./test_demo/imgs', type=str, help='testing data path')
parser.add_argument('-o','--save_path', default='./demo_seg', type=str, help='segmentation output path')
parser.add_argument('-d','--docker_folder_path', default='./team_docker', type=str, help='team docker path')
args = parser.parse_args()

test_img_path = args.test_img_path
save_path = args.save_path
docker_path = args.docker_folder_path

input_temp = './inputs/'
output_temp = './outputs'
os.makedirs(save_path, exist_ok=True)

dockers = sorted(os.listdir(docker_path))
test_cases = sorted(os.listdir(test_img_path))

for docker in dockers:
    try:
        # create temp folers for inference one-by-one
        if os.path.exists(input_temp):
            shutil.rmtree(input_temp)
        if os.path.exists(output_temp):
            shutil.rmtree(output_temp)
        os.makedirs(input_temp)
        os.makedirs(output_temp)
        # load docker and create a new folder to save segmentation results
        teamname = docker.split('.')[0].lower()
        print('teamname docker: ', docker)
        os.system('docker image load -i {}'.format(join(docker_path, docker)))
        team_outpath = join(save_path, teamname)
        if os.path.exists(team_outpath):
            shutil.rmtree(team_outpath)
        os.mkdir(team_outpath)
        os.system('chmod -R 777 ./* ')
        metric = OrderedDict()
        metric['CaseName'] = []
        metric['RunningTime'] = []
        # To obtain the running time for each case, testing cases are inferred one-by-one
        for case in test_cases:
            shutil.copy(join(test_img_path, case), input_temp)
            cmd = 'docker container run -m 8G --name {} --rm -v $PWD/inputs/:/workspace/inputs/ -v $PWD/outputs/:/workspace/outputs/ {}:latest /bin/bash -c "sh predict.sh" '.format(teamname, teamname)
            print(teamname, ' docker command:', cmd, '\n', 'testing image name:', case)
            start_time = time.time()
            os.system(cmd)
            real_running_time = time.time() - start_time
            print(f"{case} finished! Inference time: {real_running_time}")
            # save metrics
            metric['CaseName'].append(case)
            metric['RunningTime'].append(real_running_time)
            os.remove(join(input_temp, case))  
            seg_name = case
            try:
                os.rename(join(output_temp, seg_name), join(team_outpath, seg_name))
            except:
                print(f"{join(output_temp, seg_name)}, {join(team_outpath, seg_name)}")
                print("Wrong segmentation name!!! It should be the same as image_name")
        metric_df = pd.DataFrame(metric)
        metric_df.to_csv(join(team_outpath, teamname + '_running_time.csv'), index=False)
        torch.cuda.empty_cache()
        os.system("docker rmi {}:latest".format(teamname))
        shutil.rmtree(input_temp)
        shutil.rmtree(output_temp)
    except Exception as e:
        print(e)

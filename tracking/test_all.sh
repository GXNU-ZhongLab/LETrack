#!/bin/bash  
# python  tracking/train.py --script mfi_track --config vitb_256_mfit_32x1_1e2_lasher_15ep_sot_rgbt --save_dir ./output/vitb_256_mfit_32x1_1e2_lasher_15ep_sot_rgbt --mode multiple --nproc_per_node 2


for ((runid=298;runid>295;runid--))
do
    python tracking/test.py letrack baseline_WOCE --dataset_name visdrone --threads 16 --num_gpus 8 --runid $runid  && echo "Command for runid $runid executed successfully"
    python tracking/test.py letrack baseline_WOCE --dataset_name uav123 --threads 16 --num_gpus 8 --runid $runid  && echo "Command for runid $runid executed successfully"
    python tracking/test.py letrack baseline_WOCE --dataset_name uavdt --threads 16 --num_gpus 8 --runid $runid  && echo "Command for runid $runid executed successfully"
    python tracking/test.py letrack baseline_WOCE --dataset_name uavtrack112 --threads 16 --num_gpus 8 --runid $runid  && echo "Command for runid $runid executed successfully"
    python tracking/test.py letrack baseline_WOCE --dataset_name dtb70 --threads 16 --num_gpus 8 --runid $runid  && echo "Command for runid $runid executed successfully"
    python tracking/test.py letrack baseline_WOCE --dataset_name uav123_10fps --threads 16 --num_gpus 8 --runid $runid  && echo "Command for runid $runid executed successfully"
    python tracking/test.py letrack baseline_WOCE --dataset_name uavtrack --threads 16 --num_gpus 8 --runid $runid  && echo "Command for runid $runid executed successfully"
done





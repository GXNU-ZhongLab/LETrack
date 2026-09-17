# [CVPR'2026] - LETrack

The official implementation for the **CVPR 2026** paper

\[[_Toward Low-Cost yet Effective Temporal Learning for UAV Tracking_](https://openaccess.thecvf.com/content/CVPR2026/html/Xue_Toward_Low-Cost_yet_Effective_Temporal_Learning_for_UAV_Tracking_CVPR_2026_paper.html)\]

Models, Raw Results, and Training Logs are available for download at [here](https://pan.baidu.com/s/1ziHNb6ZHmEC_tlx7d-q7rw?pwd=6n35), or use [google drive](https://drive.google.com/drive/folders/1NzD3LRCr-X80qQlvs9ygcEezh_E3Gr0A?usp=drive_link)


## Training Data Preparation
Put the training datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```

## Test Data Preparation

For ease of testing, we have made the structured dataset available for download at [here](https://pan.baidu.com/s/1p0H_hHGUAc3fWkD3wlfNcw?pwd=e22r), code: e22r. 

Put the test datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- UAV123
            |-- anno
            |-- data_seq
        -- UAV123_10fps
            |-- anno
            |-- data_seq
        -- uavdt
            |-- anno
            |-- sequences
        -- V4RFlight112
            |-- anno
            |-- anno_l
            |-- data_seq
            |-- attributes
        -- DTB70
            |-- Animal1
            |-- Animal2
            ...
        -- VisDrone2018-SOT-test-dev
            |-- annotations
            |-- sequences
            |-- attributes
   ```

## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```


## Training
Download pre-trained [DeiT-tiny distilled weights](https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth) and rename as deit_distilled.pth and put it under `$PROJECT_ROOT$/pretrained_models` 

```
python tracking/train.py \
--script letrack --config baseline_WOCE \
--save_dir ./output \
--mode multiple --nproc_per_node 4 \
--use_wandb 0
```

Replace `--config` with the desired model config under `experiments/letrack`.

We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`.


## Test and Evaluation
- UAV123 or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py --tracker_param letrack --dataset uav123 --threads 8 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```
- uav123_10fps
```
python tracking/test.py  --tracker_param letrack --dataset uav123_10fps --threads 8 --num_gpus 4
```
- uavtrack_L
```
python tracking/test.py  --tracker_param letrack --dataset uavtrack --threads 8 --num_gpus 4
```
- uavtrack112
```
python tracking/test.py  --tracker_param letrack --dataset uavtrack112 --threads 8 --num_gpus 4
```
- uavdt
```
python tracking/test.py  --tracker_param letrack --dataset uavdt --threads 8 --num_gpus 4
```
- dtb70
```
python tracking/test.py  --tracker_param letrack --dataset dtb70 --threads 8 --num_gpus 4
```
- visdrone
```
python tracking/test.py  --tracker_param letrack --dataset visdrone --threads 8 --num_gpus 4
```

## Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single RTX2080Ti GPU.

```
python tracking/profile_model.py
```


## Contact
For any questions or cooperation, please contact xcc23cg@163.com or wechat: chaocan23

## Citation
If our work is useful for your research, please consider citing:

```Bibtex
@inproceedings{letrack,
  title={Toward Low-Cost yet Effective Temporal Learning for UAV Tracking},
  author={Xue, Chaocan and Liang, Qihua and Zhong, Bineng and Zu, Yanting and Xue, Yuanliang and Xia, Haiying and Song, Shuxiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={42538--42548},
  year={2026}
}
```

Friendly link: [SGLATrack (CVPR 2025)](https://github.com/GXNU-ZhongLab/SGLATrack)

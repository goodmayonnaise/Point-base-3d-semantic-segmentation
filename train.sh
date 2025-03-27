






CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" OMP_NUM_THREADS=4 \
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 --master_addr localhost main.py

CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12344 --master_addr localhost main_scunet.py 
CUDA_VISIBLE_DEVICES="4,5,6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 --master_addr localhost main_scunet.py 


CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12344 --master_addr localhost main_scunet.py 


CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 --master_addr localhost main_scunet_255.py
CUDA_VISIBLE_DEVICES="4,5,6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12346 --master_addr localhost main_scunet_norm.py
CUDA_VISIBLE_DEVICES="4,5,6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 4 main_scunet.py 



CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 --master_addr localhost main_scunet_255.py
CUDA_VISIBLE_DEVICES="2,3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12346 --master_addr localhost main_scunet_255.py
CUDA_VISIBLE_DEVICES="4,5" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12347 --master_addr localhost main_scunet_255.py
CUDA_VISIBLE_DEVICES="6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12348 --master_addr localhost main_scunet_discriminator.py

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 --master_addr localhost main_fusion.py


CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 --master_addr localhost main_fusion_resume.py

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 --master_addr localhost main_fusion.py


CUDA_VISIBLE_DEVICES="4" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12349 --master_addr localhost inference_fusion3.py
CUDA_VISIBLE_DEVICES="5,6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 3 --master_port 12347 --master_addr localhost inference_fusion4.py

CUDA_VISIBLE_DEVICES="4,5,6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12346 --master_addr localhost main_fusion6.py
CUDA_VISIBLE_DEVICES="4,5,6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12346 --master_addr localhost main_fusion6.py


CUDA_VISIBLE_DEVICES="6" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12347 --master_addr localhost inference_fusion5.py
CUDA_VISIBLE_DEVICES="7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12348 --master_addr localhost inference_fusion6.py


# 0918
CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 --master_addr localhost main_scunet0915.py
CUDA_VISIBLE_DEVICES="2,3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12346 --master_addr localhost main_scunet_org.py

CUDA_VISIBLE_DEVICES="4,5" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12347 --master_addr localhost main_adapter.py
CUDA_VISIBLE_DEVICES="6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12348 --master_addr localhost main_adapter.py


CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 --master_addr localhost main_adapter.py

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 --master_addr localhost main_adapter.py


# 0925
CUDA_VISIBLE_DEVICES="4" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12350 --master_addr localhost inference_scunet.py
CUDA_VISIBLE_DEVICES="5" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12351 --master_addr localhost inference_scunet.py
CUDA_VISIBLE_DEVICES="6" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12347 --master_addr localhost inference_scunet.py
CUDA_VISIBLE_DEVICES="7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12348 --master_addr localhost inference_scunet.py

# 0925-2 
CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 --master_addr localhost main_adapter_prev.py
CUDA_VISIBLE_DEVICES="2,3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12346 --master_addr localhost main_adapter_prev2.py
CUDA_VISIBLE_DEVICES="4,5" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12347 --master_addr localhost main_adapter_prev2.py
CUDA_VISIBLE_DEVICES="6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12348 --master_addr localhost main_adapter.py

# 0926
CUDA_VISIBLE_DEVICES="5" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12352 --master_addr localhost inference_adapter_prev.py
CUDA_VISIBLE_DEVICES="6" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12351 --master_addr localhost inference_adapter_prev.py
CUDA_VISIBLE_DEVICES="7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12350 --master_addr localhost inference_adapter_prev.py

# 0927
CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 --master_addr localhost main_adapter_prev.py
CUDA_VISIBLE_DEVICES="2,3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12346 --master_addr localhost main_adapter_prev2.py
CUDA_VISIBLE_DEVICES="4,5" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12347 --master_addr localhost main_adapter_prev3.py
CUDA_VISIBLE_DEVICES="6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12348 --master_addr localhost main_adapter_prev4.py

# 1005
CUDA_VISIBLE_DEVICES="1" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12350 --master_addr localhost inference_adapter_prev.py

# 1010
CUDA_VISIBLE_DEVICES="1" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12350 --master_addr localhost inference_adapter_1024_half.py
CUDA_VISIBLE_DEVICES="1" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12351 --master_addr localhost inference_adapter_1024.py
CUDA_VISIBLE_DEVICES="4" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12352 --master_addr localhost inference_adapter_1025_half.py
CUDA_VISIBLE_DEVICES="1" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12353 --master_addr localhost inference_adapter_1025.py

CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12345 --master_addr localhost main_adapter_depth.py


CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 --master_addr localhost main_adapter_1024.py
CUDA_VISIBLE_DEVICES="4,5,6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12346 --master_addr localhost main_generator.py
CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12347 --master_addr localhost inference_adapter_1024.py
CUDA_VISIBLE_DEVICES="4,5,6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12348 --master_addr localhost inference_generator.py


CUDA_VISIBLE_DEVICES="1,2,3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 3 --master_port 12347 --master_addr localhost main_generator_ssim.py
CUDA_VISIBLE_DEVICES="5,6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 3 --master_port 12348 --master_addr localhost main_generator_ssim.py
CUDA_VISIBLE_DEVICES="1,2,3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 3 --master_port 12349 --master_addr localhost main_generator_ssim.py
CUDA_VISIBLE_DEVICES="5,6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 3 --master_port 12355 --master_addr localhost main_generator_ssim.py

CUDA_VISIBLE_DEVICES="6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12348 --master_addr localhost main_generator_msl1.py

CUDA_VISIBLE_DEVICES="3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12352 --master_addr localhost main_generator_ms.py

CUDA_VISIBLE_DEVICES="1,3,5" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 3 --master_port 12352 --master_addr localhost main_generator_ms.py

CUDA_VISIBLE_DEVICES="6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12353 --master_addr localhost main_generator_ssim_ps.py


CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12350 --master_addr localhost inference_generator.py
CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12351 --master_addr localhost inference_generator.py
CUDA_VISIBLE_DEVICES="2" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12360 --master_addr localhost 
CUDA_VISIBLE_DEVICES="3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12359 --master_addr localhost

CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12356 --master_addr localhost inference_generator.py

CUDA_VISIBLE_DEVICES="5,6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 3 --master_port 12349 --master_addr localhost main_generator_ssim_lpips_focal_strided_ps.py



CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12360 --master_addr localhost inference_fusion.py
CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12362 --master_addr localhost inference_fusion.py
CUDA_VISIBLE_DEVICES="0" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12363 --master_addr localhost inference_fusion.py


CUDA_VISIBLE_DEVICES="4" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12354 --master_addr localhost main_generator_transferlearning.py
CUDA_VISIBLE_DEVICES="5" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12355 --master_addr localhost main_generator_transferlearning.py
CUDA_VISIBLE_DEVICES="1" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12370 --master_addr localhost main_generator_resize.py

CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 --master_addr localhost main_generator_resize.py

CUDA_VISIBLE_DEVICES="0,1,2,3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345 --master_addr localhost main_fusion_resize.py
CUDA_VISIBLE_DEVICES="4,5,6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12346 --master_addr localhost main_fusion_resize.py
CUDA_VISIBLE_DEVICES="6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12347 --master_addr localhost main_fusion_resize2.py
CUDA_VISIBLE_DEVICES="7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 4 --master_port 12348 --master_addr localhost main_generator_resize.py



#### train shell 
CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12350 --master_addr localhost main_generator_transferlearning.py
CUDA_VISIBLE_DEVICES="2,3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12351 --master_addr localhost main_generator_transferlearning.py
CUDA_VISIBLE_DEVICES="4,5" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12352 --master_addr localhost main_generator_transferlearning.py
CUDA_VISIBLE_DEVICES="6,7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12353 --master_addr localhost main_generator_transferlearning.py

CUDA_VISIBLE_DEVICES="4" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12354 --master_addr localhost inference_fusion_all.py
CUDA_VISIBLE_DEVICES="5" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12355 --master_addr localhost inference_fusion_all.py
CUDA_VISIBLE_DEVICES="6" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12356 --master_addr localhost inference_generator_front.py
CUDA_VISIBLE_DEVICES="7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12357 --master_addr localhost inference_fusion_all.py
CUDA_VISIBLE_DEVICES="5" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12358 --master_addr localhost inference_fusion_resize.py
CUDA_VISIBLE_DEVICES="6" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12359 --master_addr localhost inference_fusion_resize.py

#### test shell
sCUDA_VISIBLE_DEVICES="7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12361 --master_addr localhost inference_fusion_knn_all.py 
CUDA_VISIBLE_DEVICES="6" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12362 --master_addr localhost inference_fusion_knn_back.py
CUDA_VISIBLE_DEVICES="5" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12363 --master_addr localhost inference_fusion_knn_front.py
CUDA_VISIBLE_DEVICES="3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12364 --master_addr localhost inference_generator_all_pj.py
CUDA_VISIBLE_DEVICES="2" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12365 --master_addr localhost inference_fusion_knn_side.py


CUDA_VISIBLE_DEVICES="2" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12361 --master_addr localhost inference_generator_all_pj_360_384x6250_1311.py 

# test shell 
CUDA_VISIBLE_DEVICES="7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12361 --master_addr localhost inference_1311.py 
CUDA_VISIBLE_DEVICES="6" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12362 --master_addr localhost inference_base_aug_pad.py 
CUDA_VISIBLE_DEVICES="5" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12363 --master_addr localhost inference_1311_split.py 
CUDA_VISIBLE_DEVICES="4" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12364 --master_addr localhost inference_base_aug_pad_split.py

CUDA_VISIBLE_DEVICES="7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12365 --master_addr localhost inference_1311.py 
CUDA_VISIBLE_DEVICES="6" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12366 --master_addr localhost inference_1311.py 
CUDA_VISIBLE_DEVICES="5" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12367 --master_addr localhost inference_1311_split.py 
CUDA_VISIBLE_DEVICES="4" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12368 --master_addr localhost inference_1311_split.py 



# train shell 240527 
# docker 
CUDA_VISIBLE_DEVICES="0,1" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12350 --master_addr localhost main_generator_transferlearning.py
CUDA_VISIBLE_DEVICES="2" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12351 --master_addr localhost main_generator_transferlearning.py
CUDA_VISIBLE_DEVICES="3" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 2 --master_port 12352 --master_addr localhost main_generator_transferlearning.py
# a100
CUDA_VISIBLE_DEVICES="4" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12353 --master_addr localhost main_spconv.py
CUDA_VISIBLE_DEVICES="5" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12354 --master_addr localhost main_spconv.py

# test shell 240527
CUDA_VISIBLE_DEVICES="7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12361 --master_addr localhost 0603_generator_notsync_val.py
CUDA_VISIBLE_DEVICES="7" OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node 1 --master_port 12362 --master_addr localhost 0603_generator_notsync_val.py
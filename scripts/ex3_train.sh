set -ex

## bilinear with rmse basic with more filters
#python ../train.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix our485 --name ex3tr1_bilinear_rmse --gpu_ids 1 --model autoencoder \
#--network_name unet_bilinear_ups --n_filters 32 --shuffle_data --batch_size 64 --cudnn_benchmark --logging --n_epochs 300 --lr 0.00001 --l2_decay 0.0 --lambda_rmse 1. --lambda_ssim 0. --dim 320
#
## bilinear with rmse and ssim with more filters
#python ../train.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix our485 --name ex3tr2_bilinear_rmse_ssim --gpu_ids 1 --model autoencoder \
#--network_name unet_bilinear_ups --n_filters 32 --shuffle_data --batch_size 64 --cudnn_benchmark --logging --n_epochs 300 --lr 0.00001 --l2_decay 0.0 --lambda_rmse 1. --lambda_ssim 1. --dim 320

## tconv with rmse and 32 filters
#python ../train.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix our485 --name ex3tr3_tconv_rmse --gpu_ids 1 --model autoencoder \
#--network_name unet_transpose_conv --n_filters 32 --shuffle_data --batch_size 64 --cudnn_benchmark --logging --n_epochs 500 --lr 0.00001 --l2_decay 0.0 --lambda_rmse 1. --lambda_ssim 0. --dim 320
#
### tconv with rmse+ssim and 32 filters
#python ../train.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix our485 --name ex3tr4_tconv_rmse_ssim --gpu_ids 1 --model autoencoder \
#--network_name unet_transpose_conv --n_filters 32 --shuffle_data --batch_size 64 --cudnn_benchmark --logging --n_epochs 500 --lr 0.00001 --l2_decay 0.0 --lambda_rmse 1. --lambda_ssim 1. --dim 320
#
## bilinear with rmse basic with more filters with lr sched
#python ../train.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix our485 --name ex3tr5_bilinear_rmse --gpu_ids 1 --model autoencoder \
#--network_name unet_bilinear_ups --n_filters 32 --shuffle_data --batch_size 64 --cudnn_benchmark --logging --n_epochs 400 --l2_decay 0.0 --lambda_rmse 1. --lambda_ssim 0. \
#--lr 0.00001 --lr_policy multistep --lr_multi_steps "200" --dim 320
#
## bilinear with rmse and ssim with more filters with lr sched
#python ../train.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix our485 --name ex3tr6_bilinear_rmse_ssim --gpu_ids 1 --model autoencoder \
#--network_name unet_bilinear_ups --n_filters 32 --shuffle_data --batch_size 64 --cudnn_benchmark --logging --n_epochs 500 --l2_decay 0.0 --lambda_rmse 1. --lambda_ssim 1. \
#--lr 0.00001 --lr_policy multistep --lr_multi_steps "100, 400" --dim 320
#
## tconv with rmse and 32 filters with lr sched
#python ../train.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix our485 --name ex3tr7_tconv_rmse --gpu_ids 1 --model autoencoder \
#--network_name unet_transpose_conv --n_filters 32 --shuffle_data --batch_size 64 --cudnn_benchmark --logging --n_epochs 400 --l2_decay 0.0 --lambda_rmse 1. \
#--lr 0.00001 --lr_policy multistep --lr_multi_steps "200" --lambda_ssim 0. --dim 320

# tconv with rmse+ssim and 32 filters with lr sched
python ../train.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix our485 --name ex3tr8_tconv_rmse_ssim --gpu_ids 1 --model autoencoder \
--network_name unet_transpose_conv --n_filters 32 --shuffle_data --batch_size 128 --cudnn_benchmark --logging --n_epochs 600 --l2_decay 0.0 \
--lr 0.00001 --lr_policy multistep --lr_multi_steps "100" --lambda_rmse 1. --lambda_ssim 1. --dim 320

# tconv with rmse+ssim and 32 filters with lr sched at 300
#python ../train.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix our485 --name ex3tr9_tconv_rmse_ssim --gpu_ids 1 --model autoencoder \
#--network_name unet_transpose_conv --n_filters 32 --shuffle_data --batch_size 64 --cudnn_benchmark --logging --n_epochs 500 --l2_decay 0.0 \
#--lr 0.00001 --lr_policy multistep --lr_multi_steps "300" --lambda_rmse 1. --lambda_ssim 1. --dim 320

## tconv with rmse and 32 filters higher starting lr
python ../train.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix our485 --name ex3tr10_tconv_rmse --gpu_ids 1 --model autoencoder \
--network_name unet_transpose_conv --n_filters 32 --shuffle_data --batch_size 128 --cudnn_benchmark --logging --n_epochs 400 --l2_decay 0.0 --lambda_rmse 1. \
--lr 0.0001 --lr_policy multistep --lr_multi_steps "100" --lambda_ssim 0. --dim 320

## bilinear with rmse and ssim with more filters and a lower lr: too low lr
python ../train.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix our485 --name ex3tr11_bilinear_rmse_ssim --gpu_ids 1 --model autoencoder \
--network_name unet_bilinear_ups --n_filters 32 --shuffle_data --batch_size 128 --cudnn_benchmark --logging --n_epochs 500 --l2_decay 0.0 --lambda_rmse 1. --lambda_ssim 1. \
--lr 0.000001 --dim 320

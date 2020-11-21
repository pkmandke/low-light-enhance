set -ex

#python ../test.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix eval15 --name ex3tr1_bilinear_rmse --gpu_ids 1 --model autoencoder \
#--network_name unet_bilinear_ups --n_filters 32 --cudnn_benchmark --dim 320 --epoch 300
#
#python ../test.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix eval15 --name ex3tr2_bilinear_rmse_ssim --gpu_ids 1 --model autoencoder \
#--network_name unet_bilinear_ups --n_filters 32 --cudnn_benchmark --dim 320 --epoch 300

python ../test.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix eval15 --name ex3tr4_tconv_rmse_ssim --gpu_ids 1 --model autoencoder \
--network_name unet_transpose_conv --n_filters 32 --cudnn_benchmark --dim 320 --epoch 500

#python ../test.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix eval15 --name ex3tr8_tconv_rmse_ssim --gpu_ids 1 --model autoencoder \
#--network_name unet_transpose_conv --n_filters 32 --cudnn_benchmark --dim 320 --epoch 400

#python ../test.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix eval15 --name ex3tr5_bilinear_rmse --gpu_ids 1 --model autoencoder \
#--network_name unet_bilinear_ups --n_filters 32 --cudnn_benchmark --dim 320 --epoch 400

#python ../test.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix eval15 --name ex3tr6_bilinear_rmse_ssim --gpu_ids 1 --model autoencoder \
#--network_name unet_bilinear_ups --n_filters 32 --cudnn_benchmark --dim 320 --epoch 500

#python ../test.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix eval15 --name ex3tr3_tconv_rmse --gpu_ids 1 --model autoencoder \
#--network_name unet_transpose_conv --n_filters 32 --cudnn_benchmark --dim 320 --epoch 500

#python ../test.py --dataroot /home/pkmandke/projects/cv_project/datasets/lol --data_path_suffix eval15 --name ex3tr10_tconv_rmse --gpu_ids 1 --model autoencoder \
#--network_name unet_transpose_conv --n_filters 32 --cudnn_benchmark --dim 320 --epoch 400

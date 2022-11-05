#CUDA_VISIBLE_DEVICES=0 sh test.sh
# python test.py /home/data3/changhao/Datasets/FGVC2022_Fungi/DF21-images-300/DF21_300 \
#               --model swin_large_patch4_window12_384 \
#               --img-size 384 \
#               --num-classes 1604 \
#               --checkpoint /home/data1/lkd/Fungi_data/output/Swin-TF/freeze_layer_3/20220328-034514-swin_large_patch4_window12_384-384/Best_f1_score.pth.tar \
#               --output_path ./result/Swin_384_freeze3_f1.csv

# CUDA_VISIBLE_DEVICES=2 sh test.sh
# python test.py /home/data3/changhao/Datasets/FGVC2022_Fungi/DF21-images-300/DF21_300 \
#               --model swin_large_patch4_window12_384 \
#               --img-size 384 \
#               --num-classes 1604 \
#               --checkpoint /home/data1/lkd/Fungi_data/output/Swin-TF/freeze_layer_3/20220328-034514-swin_large_patch4_window12_384-384/Best_Top1-ACC.pth.tar \
#               --output_path ./result/Swin_384_freeze3_acc.csv \
              

# python test.py /home/data3/changhao/Datasets/FGVC2022_Fungi/DF21-images-300/DF21_300 \
#               --model swin_large_patch4_window12_384 \
#               --img-size 384 \
#               --num-classes 1604 \
#               --checkpoint /home/data1/lkd/Fungi_data/output/Swin-TF/freeze_layer_2/20220328-035127-swin_large_patch4_window12_384-384/Best_f1_score.pth.tar \
#               --output_path ./result/Swin_384_freeze2_f1.csv

#CUDA_VISIBLE_DEVICES=2 sh test.sh
#python test.py /home/data3/changhao/Datasets/FGVC2022_Fungi/DF21-images-300/DF21_300 \
#              --model swin_large_patch4_window12_384 \
#              --img-size 384 \
#              --num-classes 1604 \
#              --checkpoint /home/data1/lkd/Fungi_data/output/Swin-TF/Aug/All_aug/20220417-203543-swin_large_patch4_window12_384-384/Best_Top1-ACC.pth.tar \
#              --output_path ./result/Swin_large_384_f1.csv \
#              --scoreoutput_path ./score_result/Swin_large_384_f1.csv \
#              --batch-size 256 \
#              --crop-pct 1.0


# 300
#/home/data1/lkd/Fungi_data/output/Swin-TF/Aug/All_aug/20220417-203543-swin_large_patch4_window12_384-384/Best_Top1-ACC.pth.tar
#/home/data1/lkd/Fungi_data/output/All_aug/Swin-base/20220422-030554-swin_base_patch4_window12_384-384/Best_Top1-ACC.pth.tar
# /data/home-ustc/lkd22/FGVC_Fungi/DF21-images/DF21  0-3382562384.JPG  mode 1
#python test.py /home/data3/changhao/Datasets/FGVC2022_Fungi/DF21-images-300/DF21_300 \
#             --model swin_base_patch4_window12_384 \
#             --img-size 384 \
#             --num-classes 1604 \
#             --checkpoint /home/data1/lkd/Fungi_data/output/All_aug/Swin-base/20220422-030554-swin_base_patch4_window12_384-384/Best_Top1-ACC.pth.tar \
#             --output_path ./result/testswin_base_patch4_window12_384_ns_freeze2_Crop10_acc.csv \
#             --scoreoutput_path ./score_result/testscore_swin_base_patch4_window12_384_freeze2_Crop10_acc.csv \
#             --batch-size 16 \
#             --crop-pct 1.0 \
#             --mode 1
#
### /data/home-ustc/lkd22/FGVC_Fungi/DF20-images/DF20  2238534874-102933.JPG mode 0
#python test.py /home/data3/changhao/Datasets/FGVC2022_Fungi/DF20-300px/DF20_300 \
#              --model swin_base_patch4_window12_384 \
#              --img-size 384 \
#              --num-classes 1604 \
#              --checkpoint /home/data1/lkd/Fungi_data/output/All_aug/Swin-base/20220422-030554-swin_base_patch4_window12_384-384/Best_Top1-ACC.pth.tar \
#              --output_path ./result/trainswin_base_patch4_window12_384_freeze2_Crop10_acc.csv \
#              --scoreoutput_path ./score_result/trainscore_swin_base_patch4_window12_384_freeze2_Crop10_acc.csv \
#              --batch-size 16 \
#              --crop-pct 1.0 \
#              --mode 0





#
# /data/home-ustc/lkd22/FGVC_Fungi/DF21-images/DF21  0-3382562384.JPG  mode 1
#python test.py /home/data1/lkd/Fungi_data/Fungidata/DF21 \
#             --model swin_base_patch4_window12_384 \
#             --img-size 384 \
#             --num-classes 1604 \
#             --checkpoint /home/data1/lkd/Fungi_data/output/All_aug/Swin-base/20220422-030554-swin_base_patch4_window12_384-384/Best_Top1-ACC.pth.tar \
#             --output_path ./result/testswin_base_patch4_window12_384_ns_freeze2_Crop10_acc.csv \
#             --scoreoutput_path ./score_result/testscore_swin_base_patch4_window12_384_freeze2_Crop10_acc.csv \
#             --batch-size 16 \
#             --crop-pct 1.0 \
#             --mode 1
#
## /data/home-ustc/lkd22/FGVC_Fungi/DF20-images/DF20  2238534874-102933.JPG mode 0
#python test.py /home/data1/lkd/Fungi_data/Fungidata/DF20 \
#              --model swin_base_patch4_window12_384 \
#              --img-size 384 \
#              --num-classes 1604 \
#              --checkpoint /home/data1/lkd/Fungi_data/output/All_aug/Swin-base/20220422-030554-swin_base_patch4_window12_384-384/Best_Top1-ACC.pth.tar \
#              --output_path ./result/trainswin_base_patch4_window12_384_freeze2_Crop10_acc.csv \
#              --scoreoutput_path ./score_result/trainscore_swin_base_patch4_window12_384_freeze2_Crop10_acc.csv \
#              --batch-size 16 \
#              --crop-pct 1.0 \
#              --mode 0


#python test.py /home/data1/lkd/Fungi_data/Fungidata/DF21 \
#             --model swin_large_patch4_window12_384 \
#             --img-size 384 \
#             --num-classes 1604 \
#             --checkpoint /home/data1/lkd/Fungi_data/output/Swin-TF/DF20/All_data/swin_large_384/20220503-104748-swin_large_patch4_window12_384-384/checkpoint-23.pth.tar \
#             --output_path ./result/testswin_large_patch4_window12_384_freeze2_Crop10_acc24.csv \
#             --scoreoutput_path ./score_result/testscore_swin_large_patch4_window12_384_freeze2_Crop10_acc24.csv \
#             --batch-size 16 \
#             --crop-pct 1.0 \
#             --mode 1
#
#python test.py /home/data1/lkd/Fungi_data/Fungidata/DF21 \
#             --model swin_large_patch4_window12_384 \
#             --img-size 384 \
#             --num-classes 1604 \
#             --checkpoint /home/data1/lkd/Fungi_data/output/Swin-TF/DF20/All_data/swin_large_384/20220503-104748-swin_large_patch4_window12_384-384/checkpoint-20.pth.tar \
#             --output_path ./result/testswin_large_patch4_window12_384_freeze2_Crop10_acc21.csv \
#             --scoreoutput_path ./score_result/testscore_swin_large_patch4_window12_384_freeze2_Crop10_acc21.csv \
#             --batch-size 16 \
#             --crop-pct 1.0 \
#             --mode 1

#python test.py /home/data1/lkd/Fungi_data/Fungidata/DF21 \
#             --model swin_large_patch4_window12_384 \
#             --img-size 384 \
#             --num-classes 1604 \
#             --checkpoint /home/data1/lkd/Fungi_data/output/Swin-TF/DF20/All_aug/freeze_layer_2/20220502-114033-swin_large_patch4_window12_384-384/Best_Top1-ACC.pth.tar \
#             --output_path ./result/testswin_large_patch4_window12_384_freeze2_Crop10_acc.csv \
#             --scoreoutput_path ./score_result/testscore_swin_large_patch4_window12_384_freeze2_Crop10_acc.csv \
#             --batch-size 16 \
#             --crop-pct 1.0 \
#             --mode 1




python test.py /home/data1/lkd/Fungi_data/Fungidata/DF21 \
             --model swin_base_patch4_window12_384 \
             --img-size 384 \
             --num-classes 1604 \
             --checkpoint /home/data1/lkd/Fungi_data/Fungidata/linshi_model/swin_base_extra2_data1_largelr/checkpoint-4.pth.tar  \
             --output_path ./result/Swin_base/balance_nocrop_1.csv \
             --scoreoutput_path ./score_result/Swin_base/balance_nocrop_1.csv \
             --batch-size 32 \
             --crop-pct 1.0





python test.py /home/data1/lkd/Fungi_data/Fungidata/DF21 \
             --model swin_large_patch4_window12_384 \
             --img-size 384 \
             --num-classes 1604 \
             --checkpoint ./model/swin_large/checkpoint-4.pth.tar  \
             --output_path ./result/Swin_large/balance_nocrop_3.csv \
             --scoreoutput_path ./score_result/Swin_large/balance_nocrop_3.csv \
             --batch-size 16 \
             --crop-pct 1.0






python test.py /home/data1/lkd/Fungi_data/Fungidata/DF21 \
             --model tf_efficientnet_b6_ns \
             --num-classes 1604 \
             --checkpoint ./model/eff_b6/checkpoint-4.pth.tar  \
             --output_path ./result/eff_b6/balance_nocrop_3.csv \
             --scoreoutput_path ./score_result/eff_b6/balance_nocrop_3.csv \
             --batch-size 8 \
             --crop-pct 1.0





python test.py /home/data1/lkd/Fungi_data/Fungidata/DF21 \
             --model tf_efficientnet_b7_ns \
             --num-classes 1604 \
             --checkpoint ./model/eff_b7/checkpoint-4.pth.tar  \
             --output_path ./result/eff_b7/balance_nocrop_3.csv \
             --scoreoutput_path ./score_result/eff_b7/balance_nocrop_3.csv \
             --batch-size 8 \
             --crop-pct 1.0
             
             
 
python test.py /home/data1/lkd/Fungi_data/Fungidata/DF21 \
             --model beit_large_patch16_512 \
             --img-size 512 \
             --num-classes 1604 \
             --checkpoint ./model/beit/checkpoint-3.pth.tar  \
             --output_path ./result/beit/balance_nocrop_3.csv \
             --scoreoutput_path ./score_result/beit/balance_nocrop_3.csv \
             --batch-size 4 \
             --crop-pct 1.0

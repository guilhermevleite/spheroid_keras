EXP_NAME=0_tunet_ours_filtered_100ep
python val.py --name $EXP_NAME

EXP_NAME=2_multires_ours_filtered_100ep
python val.py --name $EXP_NAME

EXP_NAME=2_unet_ours_filtered_100ep
python val.py --name $EXP_NAME

EXP_NAME=2_unetpp_ours_filtered_100ep
python val.py --name $EXP_NAME

EXP_NAME=2_unext_ours_filtered_100ep
python val.py --name $EXP_NAME
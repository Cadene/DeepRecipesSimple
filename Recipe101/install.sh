#!/usr/bin/env bash

# download python script from OverFeat repository
wget https://raw.githubusercontent.com/sermanet/OverFeat/master/download_weights.py

# run python script to download OverFeat weights
python download_weights.py

# move weights to current directory
mv data/default/net_weight_* . 
rm -rf data/

# build C wrapper library
make clean && make

# done!
echo "==> Install completed, execute 'artistic.lua'"

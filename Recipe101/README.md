# Deep Learning with Torch7

## How do I lunch an experiment with Overfeat ?

You have to download and compile first
```
sh ./install.sh
```

Then you lunch with nohup (doesn't matter if your ssh is closed)
```
gpu=4
exp=1
name="overfeat_exp${exp}_gpu${gpu}"
path2save="./rslt/$name"
mkdir $path2save
echo "CUDA_VISIBLE_DEVICES=$gpu th -i main.lua \
-path2save $path2save \
-lr 3e-1 \
-lrd 5e-4 \
-wd 1e-3 \
-m 0.6 \
-lrf_conv 10 \
-pretrain 1" > $name.sh
chmod 777 $name.sh
cp $name.sh $path2save/$name.sh
cat $name.sh >> rslt/experiences.log
nohup ./$name.sh > $path2save/run.log &
cat $path2save/run.log
```

## Sources

https://github.com/jhjin/overfeat-torch


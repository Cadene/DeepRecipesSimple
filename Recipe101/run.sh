gpu=0
exp=5
name="exp${exp}_overfeat_${gpu}"
path2save="./rslt/$name"
mkdir $path2save
echo "CUDA_VISIBLE_DEVICES=$gpu th -i main.lua \
-path2save $path2save \
-lr 3e-1 \
-lrd 5e-4 \
-wd 1e-3 \
-m 0.6 \
-lrf_conv 1" > $name.sh
chmod 777 $name.sh
cp $name.sh $path2save/$name.sh
cat $name.sh >> rslt/experiences.log
nohup ./$name.sh > $path2save/run.log &
cat $path2save/run.log
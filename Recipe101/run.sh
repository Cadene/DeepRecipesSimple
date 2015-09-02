gpu=0
name="overfeat_${gpu}"
path2save="./rslt/$name"
mkdir $path2save
echo "CUDA_VISIBLE_DEVICES=$gpu th -i main.lua \
-path2save $path2save \
-lr 1e-1 \
-lrd 0 \
-wd 1e-3 \
-m 0.6 \
-lrf_conv 10" > $name.sh
chmod 777 $name.sh
cp $name.sh $path2save/$name.sh
cat $name.sh >> rslt/experiences.log
nohup ./$name.sh > $path2save/run.log &
cat $path2save/run.log

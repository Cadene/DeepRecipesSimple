require 'image'

timer = torch.Timer()

torch.manualSeed(1337)
torch.setnumthreads(4)

path2mean = 'mean_augmented.jpg'
path2std = 'mean_augmented.jpg'

mean = image.load(path2mean)
std = image.load(path2std)

path2dir = '/home/cadene/data/recipe_101_augmented/'
path2train = path2dir..'train/'
path2test = path2dir..'test/'

path2dir_n = '/home/cadene/data/recipe_101_augm_mean_std/'
path2train_n = path2dir_n..'train/'
path2test_n = path2dir_n..'test/'

os.execute('mkdir -p '..path2dir_n)
os.execute('mkdir -p '..path2train_n)
os.execute('mkdir -p '..path2test_n)

local path2esc = {'.', '..', '.DS_Store', '._.DS_Store'}
local is_in = function (string, path2esc)
    for k, name in pairs(path2esc) do
        if string == name then
            return true
        end
    end
    return false
end

-- class directories creation
for _, class in pairs(paths.dir(path2train)) do
    if not is_in(class, path2esc) then -- rm path2esc values
       path2class_train = paths.concat(path2train_n, class)
       path2class_test = paths.concat(path2test_n, class)
       os.execute('mkdir -p '..path2class_train) 
       os.execute('mkdir -p '..path2class_test) 
    end
end

os.execute('chmod -R 777 '..path2dir_n)

timer_train = torch.Timer()
for _, class in pairs(paths.dir(path2train)) do
    if not is_in(class, path2esc) then -- rm path2esc values
        for _, path_img in pairs(paths.dir(path2train..class)) do
            if not is_in(path_img, path2esc) then
                path2img = paths.concat(path2train, class, path_img)
                path2img_n = paths.concat(path2train_n, class, path_img)
                img = image.load(path2img)
                if img:size(1) == 1 then
                    tmp = torch.zeros(3,221,221)
                    tmp[1] = img:clone()
                    tmp[2] = img:clone()
                    tmp[3] = img:clone()
                    img = tmp
                end
                img:add(-mean)
                img:cdiv(std)
                image.save(path2img_n, img)
            end
        end
        s = timer:time().real
        print('train: '..class..' done in '
            ..string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60))
        timer:reset()
    end
end
s = timer_train:time().real
print('Train done in '
    ..string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60))

timer_test = torch.Timer()
for _, class in pairs(paths.dir(path2test)) do
    if not is_in(class, path2esc) then -- rm path2esc values
        for _, path_img in pairs(paths.dir(path2test..class)) do
            if not is_in(path_img, path2esc) then
                path2img = paths.concat(path2test, class, path_img)
                path2img_n = paths.concat(path2test_n, class, path_img)
                img = image.load(path2img)
                if img:size(1) == 1 then
                    tmp = torch.zeros(3,221,221)
                    tmp[1] = img:clone()
                    tmp[2] = img:clone()
                    tmp[3] = img:clone()
                    img = tmp
                end
                img:add(-mean)
                img:cdiv(std)
                image.save(path2img_n, img)
            end
        end
        s = timer:time().real
        print('test: '..class..' done in '
            ..string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60))
        timer:reset()
    end
end
s = timer_test:time().real
print('Test done in '
    ..string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60))


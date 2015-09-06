require 'image'

torch.setnumthreads(4)

timer = torch.Timer()

load_mean = true
path2train = '/home/cadene/data/recipe_101_augmented/train/'

mean = torch.zeros(3, 221, 221)
std = torch.zeros(3, 221, 221)

local path2esc = {'.', '..', '.DS_Store', '._.DS_Store'}
local is_in = function (string, path2esc)
    for k, name in pairs(path2esc) do
        if string == name then
            return true
        end
    end
    return false
end

nb_img = 0
timer_mean = torch.Timer()
for _, class in pairs(paths.dir(path2train)) do
    if not is_in(class, path2esc) then -- rm path2esc values
        local mean_class = torch.zeros(3, 221, 221)
        for _, path_img in pairs(paths.dir(path2train..class)) do
            if not is_in(path_img, path2esc) then
                local path2img = paths.concat(path2train, class, path_img)
                local img = image.load(path2img)
                if img:size(1) == 1 then
                    tmp = torch.zeros(3,221,221)
                    tmp[1] = img:clone()
                    tmp[2] = img:clone()
                    tmp[3] = img:clone()
                    img = tmp
                end
                mean_class:add(img)
                nb_img = nb_img + 1
            end
        end
        mean:add(mean_class)
        s = timer:time().real
        print(class..' done in '
            ..string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60))
        print('mean value = '..mean[{1,1,1}]..' / '..nb_img) 
        timer:reset()
    end
end
mean = mean / nb_img
image.save('mean_augmented.jpg', mean)
s = timer_mean:time().real
print('Mean done in '
    ..string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60))

nb_img = 0
timer_std = torch.Timer()
for _, class in pairs(paths.dir(path2train)) do
    if not is_in(class, path2esc) then -- rm path2esc values
        local std_class = torch.zeros(3, 221, 221)
        for _, path_img in pairs(paths.dir(path2train..class)) do
            if not is_in(path_img, path2esc) then
                local path2img = paths.concat(path2train, class, path_img)
                local img = image.load(path2img)
                if img:size(1) == 1 then
                    tmp = torch.zeros(3,221,221)
                    tmp[1] = img:clone()
                    tmp[2] = img:clone()
                    tmp[3] = img:clone()
                    img = tmp
                end
                local tmp = img - mean
                tmp:pow(2)
                std_class:add(tmp)
                nb_img = nb_img + 1
            end
        end
        std:add(std_class)
        s = timer:time().real
        print(class..' done in '
            ..string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)) 
        print('std value = sqrt ('..std[{1,1,1}]..' / '..nb_img..' )')
        timer:reset()
    end
end
std = std / nb_img
std = std:sqrt()
s = timer_std:time().real
print('Std done in '
    ..string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60)) 
image.save('std_augmented.jpg', std)




require 'image'

timer = torch.Timer()

torch.manualSeed(1337)
torch.setnumthreads(4)

function load_recipe101(path2dir, pc_train)
    local path2esc = {'.', '..', '.DS_Store', '._.DS_Store'}
    local pc_train = pc_train or 0.8 
    local path2img, label = {}, {}
    local p2i_train, l_train = {}, {}
    local p2i_test, l_test = {}, {}
    local label2class = {}
    local is_in = function (string, path2esc)
        for k, name in pairs(path2esc) do
            if string == name then
                return true
            end
        end
        return false
    end
    for _, class in pairs(paths.dir(path2dir)) do
        if not is_in(class, path2esc) then
            path2img[class] = {}
            for _, path_img in pairs(paths.dir(path2dir..class)) do
                if not is_in(path_img, path2esc) then
                   table.insert(path2img[class], path_img)
                end
            end
        end
    end
    local label = 1
    for class, p2i in pairs(path2img) do
        label2class[label] = class
        local shuffle = torch.randperm(#p2i)
        local limit = #p2i * pc_train 
        for i = 1, #p2i do
            if i < limit then
                table.insert(p2i_train, p2i[shuffle[i]])
                table.insert(l_train, label)
            else
                table.insert(p2i_test, p2i[shuffle[i]])
                table.insert(l_test, label)
            end
        end
        label = label + 1
    end
    local trainset = {
        path  = p2i_train,
        label = l_train,
        size  = #p2i_train
    }
    local testset = {
        path  = p2i_test,
        label = l_test,
        size = #p2i_test
    }
    return trainset, testset, label2class
end

function prepare_img(path2img, dim_in, dim_out, crop_type, flip, mean, std)
    local dim     = dim_in
    local dim_out = dim_out
    local img_dim
    local img_raw = image.load(path2img) -- [0,1] -> [0,255]img
    local rh = img_raw:size(2)
    local rw = img_raw:size(3)

    -- rescale to 3 * 256 * 256
    if rh < rw then
       rw = math.floor(rw / rh * dim)
       rh = dim
    else
       rh = math.floor(rh / rw * dim)
       rw = dim
    end
    local img_scale = image.scale(img_raw, rw, rh)
    local offsetx = 1
    local offsety = 1
    if rh < rw then
        offsetx = offsetx + math.floor((rw-dim)/2)
    else
        offsety = offsety + math.floor((rh-dim)/2)
    end
    img = img_scale[{{},{offsety,offsety+dim-1},{offsetx,offsetx+dim-1}}]

    if crop_type then
        local w1, h1
        if crop_type == 1 then -- center
            w1 = math.ceil((dim - dim_out) / 2)
            h1 = math.ceil((dim - dim_out) / 2)
        elseif crop_type == 2 then -- top-left
            w1 = 1
            h1 = 1
        elseif crop_type == 3 then -- top-right
            w1 = dim - dim_out
            h1 = 1
        elseif crop_type == 4 then -- bottom-left
            w1 = 1
            h1 = dim - dim_out
        elseif crop_type == 5 then -- bottom-right
            w1 = dim - dim_out
            h1 = dim - dim_out
        else
            error('crop_type error')
        end
        img = image.crop(img, w1, h1, w1 + dim_out, h1 + dim_out)
    end
    
    if flip == 1 then
        img = image.hflip(img)
    end

    -- add mean and div std
    if mean and std then 
        img:add(mean)
        img:cdiv(std)
    end

    return img
end

path2dir  = '/home/cadene/data/recipe_101_clean/' 
realdir   = '/home/cadene/data/recipe_101/recipe_101/'
path2augm = '/home/cadene/data/recipe_101_augmented/'

trainset, testset, label2class = load_recipe101(path2dir)

path2train = paths.concat(path2augm, 'train/')
path2test  = paths.concat(path2augm, 'test/')

crop_type = {1,2,3,4,5,1,2,3,4,5}
flip      = {0,0,0,0,0,1,1,1,1,1}

os.execute('mkdir -p '..path2train)
os.execute('mkdir -p '..path2test)
os.execute('chmod -R 777 '..path2augm)

for i = 1, #trainset.path do
    class = label2class[trainset.label[i]]
    path2class = paths.concat(path2train, class)
    os.execute('mkdir -p '..path2class)
    os.execute('chmod 777 '..path2class)
    path2img_from = paths.concat(realdir, class, trainset.path[i])
    for j = 1, 10 do
        start, _ = string.find(trainset.path[i], '.jpg')
        tmp_name = string.sub(trainset.path[i], 1, start-1)
        path2img_to  = paths.concat(path2train, class, tmp_name..'_'..j..'.jpg')
        img = prepare_img(path2img_from, 256, 221, crop_type[j], flip[j])
        image.save(path2img_to, img)
    end
end

s = timer:time().real
print('Train done in '..string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60).." left"))
timer:reset()

for i = 1, #testset.path do
    class = label2class[testset.label[i]]
    path2class = paths.concat(path2test, class)
    os.execute('mkdir -p '..path2class)
    os.execute('chmod 777 '..path2class)
    path2img_from = paths.concat(realdir, class, testset.path[i])
    img = prepare_img(path2img_from, 221, 221)
    path2img_to  = paths.concat(path2train, class, testset.path[i])
    image.save(path2img_to, img)
end

s = timer:time().real
print('Test done in '..string.format("%.2d:%.2d:%.2d", s/(60*60), s/60%60, s%60).." left"))





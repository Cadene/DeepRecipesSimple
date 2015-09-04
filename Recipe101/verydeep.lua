posix = require 'posix'
require 'optim'
require 'nn'
require 'image'
require 'loadcaffe'

cmd = torch.CmdLine()
cmd:option('-path2save', 'rslt/', '')
cmd:option('-lr', 1e-1, '')
cmd:option('-lrd', 0, '')
cmd:option('-wd', 1e-3, '')
cmd:option('-m', 0.6, '')
cmd:option('-lrf_conv', 10, 'lr factor')
opt = cmd:parse(arg or {})

cuda = false
batch = 60
nb_epoch = 60
seed = 1337
-- path2dir = '/Users/remicadene/data/recipe_101_tiny/'
-- path2dir = '/home/cadene/data/recipe_101_tiny/'
path2dir = '/home/cadene/data/recipe_101_clean/'
save_model = false
if opt.pretrain == 1 then
    pretrain_model = true
else 
    pretrain_model = false
end
debug_mode = false
path2save = opt.path2save

print("# ... lunching using pid = "..posix.getpid("pid"))
torch.manualSeed(seed)
torch.setnumthreads(4)
torch.setdefaulttensortype('torch.FloatTensor')

if cuda then
    print('# ... switching to CUDA')
    require 'cutorch'
    cutorch.setDevice(1)
    cutorch.manualSeed(seed)
    require 'cunn'
    require 'cudnn'
end

function debug(...)
    local arg = {...}
    if debug_mode then
        print('DEBUG', ...)
    end
end

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

trainset, testset, label2class = load_recipe101(path2dir)
nb_class = #label2class

if cuda then
    SpatialConvolution = cudnn.SpatialConvolution
    SpatialMaxPooling  = cudnn.SpatialMaxPooling
else
    SpatialConvolution = nn.SpatialConvolutionMM
    SpatialMaxPooling  = nn.SpatialMaxPooling
end

print('# ... loading very deep')
model = loadcaffe.load('VGG_ILSVRC_16_layers_deploy.prototxt', 'VGG_ILSVRC_16_layers.caffemodel', 'cudnn')

print(model)

if cuda then model:cuda() end
print('# ... reshaping parameters and gradParameters')
parameters, gradParameters = model:getParameters()

criterion = nn.ClassNLLCriterion()
if cuda then criterion:cuda() end

confusion   = optim.ConfusionMatrix(nb_class)
trainLogger = optim.Logger(paths.concat(path2save, 'train.log'))
testLogger  = optim.Logger(paths.concat(path2save, 'test.log'))
lossLogger  = optim.Logger(paths.concat(path2save, 'loss.log'))

-- optimizer sgd
config = {
    learningRate = opt.lr,--1e-5,
    weightDecay = opt.wd,
    momentum = opt.m,
    learningRateDecay = opt.lrd
}
if opt.lrf_conv ~= 1 then
    print('# ... making learningRates for convolution layers')
    lr_conv = config.learningRate / opt.lrf_conv
    lrs = torch.Tensor(parameters:size(1))
    i = 0
    lrs:apply(function()
        i = i + 1
        if i <= 18916480 then
            return lr_conv
        else
            return config.learningRate
        end
    end)
    config.learningRates = lrs
end

function train()
    print('# ---------------------- #')
    print('# ... training model ... #')
    print('# ---------------------- #')
    collectgarbage()
    local timer = torch.Timer()
    t_outputs, t_targets = {}, {}
    model:training()
    shuffle = torch.randperm(trainset.size)
    batch_id = 1
    for i = 1, trainset.size, batch do
        print('# ... processing batch_'..batch_id)
        if i + batch > trainset.size then
            b_size = trainset.size - i
        else
            b_size = batch
        end
        inputs  = torch.zeros(b_size, 3, 221, 221)
        targets = torch.zeros(b_size)
        for j = 1, b_size do
            path2img   = paths.concat(path2dir,
                label2class[trainset.label[shuffle[i+j]]], 
                trainset.path[shuffle[i+j]])
            inputs[j]  = image.load(path2img)
            targets[j] = trainset.label[shuffle[i+j]]
        end
        if cuda then
            inputs  = inputs:cuda()
            targets = targets:cuda()
        end
        feval = function (x)
            gradParameters:zero()
            outputs = model:forward(inputs)
            loss    = criterion:forward(outputs, targets)
            df_do   = criterion:backward(outputs, targets)
            df_di   = model:backward(inputs, df_do)
            _, amax = outputs:max(2)
            table.insert(t_outputs, amax:resizeAs(targets))
            table.insert(t_targets, targets:clone())
            print('> loss : '..loss)
            print('> learning rate : '..(config.learningRate / (1 + nevals*config.learningRateDecay))) 
            lossLogger:add{['loss'] = loss}
            return loss, gradParameters
        end
        optim.sgd(feval, parameters, config)
        print('> seconds : '..timer:time().real)
    batch_id = batch_id + 1
    end
    -- print(confusion)
    confusion:zero()
    for i = 1, #t_outputs do
        confusion:batchAdd(t_outputs[i], t_targets[i])
    end
    confusion:updateValids()
    print('> perf train : '..(confusion.totalValid * 100))
    trainLogger:add{['% train perf'] = confusion.totalValid * 100}
    trainLogger:style{['% train perf'] = '-'}
    trainLogger:plot()
    lossLogger:style{['loss'] = '-'}
    lossLogger:plot()
    if save_model then
        print('# ... saving model')
        torch.save(paths.concat(path2save, 'model.t7'), model)
    end
end

function test()
    print('# --------------------- #')
    print('# ... testing model ... #')
    print('# --------------------- #')
    collectgarbage()
    local timer = torch.Timer()
    t_outputs, t_targets = {}, {}
    model:evaluate()
    local batch_id = 1
    for i = 1, testset.size, batch do
        print('# ... processing batch_'..batch_id)
    if i + batch > testset.size then
            b_size = testset.size - i
        else
            b_size = batch
        end
        inputs  = torch.zeros(b_size, 3, 221, 221)
        targets = torch.zeros(b_size)
        for j = 1, b_size do
            path2img   = paths.concat(path2dir,
                label2class[testset.label[i+j]],
                testset.path[i+j])
            inputs[j]  = image.load(path2img)
            targets[j] = testset.label[i+j]
        end
        if cuda then
            inputs  = inputs:cuda()
            targets = targets:cuda()
        end
        outputs = model:forward(inputs)
        _, amax = outputs:max(2)
        table.insert(t_outputs, amax:resizeAs(targets))
        table.insert(t_targets, targets:clone())
        print('> seconds : '..timer:time().real)
        batch_id = batch_id + 1
    end
    -- print(confusion)
    confusion:zero()
    for i = 1, #t_outputs do
        debug('t_outputs', t_outputs[i])
        debug('t_targets', t_targets[i])
    confusion:batchAdd(t_outputs[i], t_targets[i])
    end
    confusion:updateValids()
    print('> perf test : '..(confusion.totalValid * 100))
    testLogger:add{['% test perf'] = confusion.totalValid * 100}
    testLogger:style{['% test perf'] = '-'}
    testLogger:plot()
end

nevals = 1 -- same nevals from optim.sgd
for epoch_id = 1, nb_epoch do
    print('\n# # # # # # # # # # # # # # # #')
    print('   ... Processing epoch_'..epoch_id..' ...')
    print('# # # # # # # # # # # # # # # #')
    train()
    test()
    nevals = nevals + 1
end

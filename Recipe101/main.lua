posix = require 'posix'
require 'optim'
require 'nn'
require 'image'

print("# ... lunching using pid = "..posix.getpid("pid"))
torch.manualSeed(1337)
torch.setnumthreads(4)

cuda = false
batch = 60
nb_epoch = 30
path2dir = '/Users/remicadene/data/recipe_101_tiny/'

if cuda then
    print('# ... switching to CUDA')
    require 'cutorch'
    cutorch.setDevice(0)
    cutorch.manualSeed(1337, 0)
    require 'cunn'
    require 'cudnn'
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

if cuda then
    SpatialConvolution = cudnn.SpatialConvolution
    SpatialMaxPooling  = cudnn.SpatialMaxPooling
else
    SpatialConvolution = nn.SpatialConvolutionMM
    SpatialMaxPooling  = nn.SpatialMaxPooling
end

print('# ... loading overfeat')
model = nn.Sequential()
model:add(SpatialConvolution(3, 96, 7, 7, 2, 2))
model:add(nn.ReLU(true))
model:add(SpatialMaxPooling(3, 3, 3, 3))
model:add(SpatialConvolution(96, 256, 7, 7, 1, 1))
model:add(nn.ReLU(true))
model:add(SpatialMaxPooling(2, 2, 2, 2))
model:add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
model:add(SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
model:add(SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
model:add(SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
model:add(nn.ReLU(true))
-- classifier
model:add(SpatialMaxPooling(3, 3, 3, 3))
model:add(SpatialConvolution(1024, 4096, 5, 5, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.5))
model:add(SpatialConvolution(4096, 4096, 1, 1, 1, 1))
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.5))
model:add(SpatialConvolution(4096, 101, 1, 1, 1, 1))
model:add(nn.View(101))
model:add(nn.LogSoftMax())
if cuda then model:cuda() end
print('# ... reshaping parameters and gradParameters')
parameters, gradParameters = model:getParameters()

criterion = nn.ClassNLLCriterion()
if cuda then criterion:cuda() end

confusion   = optim.ConfusionMatrix(101)
trainLogger = optim.Logger(paths.concat('train.log'))
testLogger  = optim.Logger(paths.concat('test.log'))

config = {
    learningRate = 1e-2,
    weightDecay = 1e-3,
    momentum = 0.9,
    learningRateDecay = 1e-7
}

function train()
    print('\ntrain')
    model:training()
    confusion:zero()
    shuffle = torch.randperm(trainset.size)
    batch_id = 1
    for i = 1, trainset.size, batch do
        print('batch_id', batch_id)
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
            print('loss', loss)
            df_do   = criterion:backward(outputs, targets)
            df_di   = model:backward(inputs, df_do)
            confusion:batchAdd(outputs, targets)
            return loss, gradParameters
        end
        optim.sgd(feval, parameters, config)
        batch_id = batch_id + 1
    end
    -- print(confusion)
    confusion:updateValids()
    trainLogger:add{['% train perf'] = confusion.totalValid * 100}
    trainLogger:style{['% train perf'] = '-'}
    trainLogger:plot()
    torch.save('model.t7', model)
end

function test()
    print('\ntest')
    model:evaluate()
    confusion:zero()
    for i = 1, testset.size, batch do
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
        confusion:batchAdd(outputs, targets)
    end
    -- print(confusion)
    confusion:updateValids()
    testLogger:add{['% test perf'] = confusion.totalValid * 100}
    testLogger:style{['% test perf'] = '-'}
    testLogger:plot()
end

for i = 1, nb_epoch do
    train()
    test()
end

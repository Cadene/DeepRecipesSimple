posix = require 'posix'
require 'optim'
require 'nn'

print("# ... lunching using pid = "..posix.getpid("pid"))
torch.manualSeed(1337)
torch.setnumthreads(4)

-- print('# ... switching to CUDA')
-- require 'cutorch'
-- cutorch.setDevice(0)
-- cutorch.manualSeed(1337, 0)
-- require 'cunn'
-- require 'cudnn'

function generate_spiral(N, D, K)
    local X = torch.zeros(N*K, D)
    local y = torch.zeros(N*K)
    for j = 1, K do
        local r  = torch.linspace(0.0, 1, N)
        local t  = torch.linspace((j-1)*4, j*4, N) + torch.randn(N)*0.2
        local X1 = torch.sin(t)
        local X2 = torch.cos(t)
        for i = 1, N do
            xi = N*(j-1)+i
            X[{xi,1}] = r[i]*X1[i]
            X[{xi,2}] = r[i]*X2[i]
            y[{xi}] = j
        end
    end
    return X, y
end

nb_class   = 3
train_size = 50 * nb_class
test_size  = 10 * nb_class
X_train, y_train = generate_spiral(50, 2, nb_class)
X_test, y_test   = generate_spiral(10, 2, nb_class)

model = nn.Sequential()
model:add(nn.Linear(2, 100))
model:add(nn.ReLU(true))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(100, nb_class))
model:add(nn.LogSoftMax())

parameters, gradParameters = model:getParameters()

criterion = nn.ClassNLLCriterion()

confusion = optim.ConfusionMatrix(nb_class)
-- model:cuda()
-- criterion:cuda()

trainLogger = optim.Logger(paths.concat('train.log'))
testLogger = optim.Logger(paths.concat('test.log'))

config = {
    learningRate = 1e-2,
    weightDecay = 1e-3,
    momentum = 0.9,
    learningRateDecay = 1e-7
}

function train()
    model:training()
    confusion:zero()
    shuffle = torch.randperm(train_size)
    for i = 1, train_size do
        input  = X_train[shuffle[i]]
        target = y_train[shuffle[i]]

        feval = function (x)
            gradParameters:zero()
            output = model:forward(input)
            print(output, target)
            loss   = criterion:forward(output, target)
            df_do  = criterion:backward(output, target)
            df_di  = model:backward(input, df_do)
            confusion:add(output, target)
            return loss, gradParameters
        end
        optim.sgd(feval, parameters, config)
    end
    print(confusion)
    -- confusion:updateValids()
    trainLogger:add{['% train perf'] = confusion.totalValid * 100}
    trainLogger:style{['% train perf'] = '-'}
    trainLogger:plot()
    torch.save('model.t7', model)
end

function test()
    model:evaluate()
    confusion:zero()
    for i = 1, test_size do
        input  = X_test[i]
        target = y_test[i]
        output = model:forward(input)
        confusion:add(output, target)
    end
    print(confusion)
    -- confusion:updateValids()
    testLogger:add{['% test perf'] = confusion.totalValid * 100}
    testLogger:style{['% test perf'] = '-'}
    testLogger:plot()
end

--train()
--test()

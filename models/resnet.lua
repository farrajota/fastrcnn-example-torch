--[[
    Resnet (18, 32, 50, 101, 152, 200) FRCNN model.
]]

require 'nn'
require 'cudnn'
require 'inn'
inn.utils = require 'inn.utils'
local utils = require 'fastrcnn.utils'

------------------------------------------------------------------------------------------------------------

local function CreateModel(nGPU, nClasses, netType)

    assert(nGPU)
    assert(nClasses)
    assert(netType)

    local available_nets = {
        ['resnet18'] = {512, 'resnet-18'},
        ['resnet32'] = {512, 'resnet-32'},
        ['resnet50'] = {2048, 'resnet-50'},
        ['resnet101'] = {2048, 'resnet-101'},
        ['resnet152'] = {2048, 'resnet-152'},
        ['resnet200'] = {2048,'resnet-200'}
    }

    local info = available_nets[string.lower(netType)]
    assert(info, 'Invalid network: '..netType..'. Available networks: resnet18, resnet32, resnet50, resnet101, resnet152, resnet200.')

    -- load features + model parameters (mean/std,stride/num feats (last conv)/colorspace format)
    local net = torch.load(paths.concat(projectDir, 'data/pretrained_models/model_'..info[2]..'.t7'))
    local model_parameters = torch.load(paths.concat(projectDir, 'data/pretrained_models/parameters_'..info[2]..'.t7'))
    net:cuda():evaluate()
    local features = net
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())


    local input = torch.randn(1, 3, 224, 224):cuda()
    utils.model.testSurgery(input, utils.model.DisableFeatureBackprop, features, 5)
    utils.model.testSurgery(input, inn.utils.foldBatchNorm, features:findModules'nn.NoBackprop'[1])
    utils.model.testSurgery(input, inn.utils.BNtoFixed, features, true)
    utils.model.testSurgery(input, inn.utils.BNtoFixed, net, true)


    -- setup classifier
    local classifier = nn.Sequential()
        :add(nn.Linear(info[1]*7*7, 4096))
        :add(nn.ReLU(true))
        :add(nn.Dropout(0.5))
        :add(nn.Linear(4096, 4096))
        :add(nn.ReLU(true))
        :add(nn.Dropout(0.5))

    -- create model
    local model = nn.Sequential()
        :add(nn.ParallelTable()
            :add(utils.model.makeDataParallel(features, nGPU))
            :add(nn.Identity())
        )
        :add(inn.ROIPooling(7, 7, 1/32))
        :add(nn.View(-1):setNumInputDims(3))
        :add(classifier)
        :add(utils.model.CreateClassifierBBoxRegressor(4096, nClasses))

    return model, model_parameters
end

------------------------------------------------------------------------------------------------------------

return CreateModel
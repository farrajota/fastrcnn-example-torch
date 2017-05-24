--[[
    VGG (16-19) FRCNN model.
]]


require 'nn'
require 'cudnn'
require 'inn'
local utils = require 'fastrcnn.utils'

------------------------------------------------------------------------------------------------------------

local function CreateModel(nGPU, nClasses, netType)

    assert(nGPU)
    assert(nClasses)
    assert(netType)


    local function SelectFeatsDisableBackprop(net)
        local features = net
        features:remove(features:size()) -- remove logsoftmax layer
        features:remove(features:size()) -- remove 3rd linear layer
        features:remove(features:size()) -- remove 2nd dropout layer
        features:remove(features:size()) -- remove 2nd last relu layer
        features:remove(features:size()) -- remove 2nd linear layer
        features:remove(features:size()) -- remove 1st dropout layer
        features:remove(features:size()) -- remove 1st relu layer
        features:remove(features:size()) -- remove 1st linear layer
        features:remove(features:size()) -- remove view layer
        features:remove(features:size()) -- remove max pool
        utils.model.DisableFeatureBackprop(features, 10)
        return features
    end


    -- load features + model parameters (mean/std,stride/num feats (last conv)/colorspace format)
    local features, model_parameters
    if netType == 'vgg16' or netType == 'vgg' then
        local net = torch.load(paths.concat(projectDir, 'data/pretrained_models/model_vgg16.t7'))
        model_parameters = torch.load(paths.concat(projectDir, 'data/pretrained_models/parameters_vgg16.t7'))
        features = SelectFeatsDisableBackprop(net)
    elseif netType == 'vgg19' then
        local net = torch.load(paths.concat(projectDir, 'data/pretrained_models/model_vgg19.t7'))
        model_parameters = torch.load(paths.concat(projectDir, 'data/pretrained_models/parameters_vgg19.t7'))
        features = SelectFeatsDisableBackprop(net)
    else
        error('Invalid network type: '.. netType..'. Available networks: vgg16, vgg19.')
    end

    -- setup classifier
    local classifier = nn.Sequential()
        :add(nn.Linear(512*7*7, 4096))
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
        :add(inn.ROIPooling(7,7,1/16))
        :add(nn.View(-1):setNumInputDims(3))
        :add(utils.model.makeDataParallel(classifier, nGPU))
        :add(utils.model.CreateClassifierBBoxRegressor(4096, nClasses))

    return model, model_parameters
end

------------------------------------------------------------------------------------------------------------

return CreateModel
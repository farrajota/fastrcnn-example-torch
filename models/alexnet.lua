--[[
    Alexnet FRCNN model.
]]


require 'nn'
require 'cunn'
require 'cudnn'
require 'inn'
local utils = require 'fastrcnn.utils'

------------------------------------------------------------------------------------------------------------

local function CreateModel(nGPU, nClasses)

    assert(nGPU)
    assert(nClasses)

    -- load features + model parameters (mean/std,stride/num feats (last conv)/colorspace format)
    local net = torch.load(paths.concat(projectDir, 'data/pretrained_models/model_alexnet.t7'))
    local model_parameters = torch.load(paths.concat(projectDir, 'data/pretrained_models/parameters_alexnet.t7'))
    local features = net

    -- remove all unnecessary layers
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())

    -- setup classifier
    local classifier = nn.Sequential()
        :add(nn.Linear(256*6*6, 4096))
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
        :add(inn.ROIPooling(6,6,1/16))
        :add(nn.View(-1):setNumInputDims(3))
        :add(classifier)
        :add(utils.model.CreateClassifierBBoxRegressor(4096, nClasses))

    return model:cuda(), model_parameters
end

------------------------------------------------------------------------------------------------------------

return CreateModel
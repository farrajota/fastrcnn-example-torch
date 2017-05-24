--[[
    Zeiler FRCNN model.
]]


require 'nn'
require 'cudnn'
require 'inn'
local utils = require 'fastrcnn.utils'

------------------------------------------------------------------------------------------------------------

local function CreateModel(nGPU, nClasses)

    assert(nGPU)
    assert(nClasses)

    -- load features + model parameters (mean/std,stride/num feats (last conv)/colorspace format)
    local net = torch.load(paths.concat(projectDir, 'data/pretrained_models/model_zeilernet.t7'))
    local model_parameters = torch.load(paths.concat(projectDir, 'data/pretrained_models/parameters_zeilernet.t7'))
    local features = net.modules[1]

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

    return model, model_parameters
end

------------------------------------------------------------------------------------------------------------

return CreateModel
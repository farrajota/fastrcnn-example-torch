--[[
    Googlenet Inception v3 FRCNN model.
]]

require 'nn'
require 'cudnn'
require 'inn'
inn.utils = require 'inn.utils'


local function CreateModel(opt, utils)
  
    assert(opt)
    assert(utils)
    
    -- load features + model parameters (mean/std,stride/num feats (last conv)/colorspace format)
    local net = torch.load('../data/pretrained_models/model_googlenet_inceptionv3_cunn.t7'):cuda()
    
    local input = torch.randn(1,3,299,299):cuda()
    local output1 = net:forward(input):clone()
    inn.utils.BNtoFixed(net, true)
    local output2 = net:forward(input):clone()
    assert((output1 - output2):abs():max() < 1e-5)

    local features = net
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())

    utils.testSurgery(input, utils.model.DisableFeatureBackprop, features, 16)
    utils.testSurgery(input, inn.utils.foldBatchNorm, features:findModules'nn.NoBackprop'[1])
    
    -- setup classifier
    local classifier = nn.Sequential()
        :add(nn.Linear(2048*7*7, 4096))
        :add(nn.ReLU(true))
        :add(nn.Dropout(0.5))
        :add(nn.Linear(4096, 4096))
        :add(nn.ReLU(true))
        :add(nn.Dropout(0.5))
    
    -- create model
    local model = nn.Sequential()
        :add(nn.ParallelTable()
            :add(utils.model.makeDataParallel(features, opt.nGPU))
            :add(nn.Identity())
        )
        :add(inn.ROIPooling(7,7,1/37.375))
        :add(nn.View(-1):setNumInputDims(3))
        :add(classifier)
        :add(utils.model.CreateClassifierBBoxRegressor(4096, opt.nClasses, opt.has_bbox_regressor))
    
    
    local model_parameters = {
        colourspace = 'rgb',
        meanstd = {mean = {128,128,128}, std = {0.0078125,0.0078125,0.0078125}},
        pixel_scale = 255,
        stride = 37.375
    }
    
    return model, model_parameters
end

return CreateModel
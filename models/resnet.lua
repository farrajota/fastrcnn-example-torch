--[[
    Resnet (18, 32, 50, 101, 152, 200) FRCNN model.
]]

require 'nn'
require 'cudnn'
require 'inn'
inn.utils = require 'inn.utils'


local function CreateModel(opt, utils)
  
    assert(opt)
    assert(utils)
    
    local available_nets = {
        ['resnet18'] = 512,
        ['resnet32'] = 512,
        ['resnet50'] = 2048,
        ['resnet101'] = 2048,
        ['resnet152'] = 2048,
        ['resnet200'] = 2048
    }
    
    local nfeats = available_nets[string.lower(opt.netType)]
    assert(nfeats, 'Undefined network: '..netType..'. Available networks: resnet18, resnet32, resnet50, resnet101, resnet152, resnet200.')
    
    -- load features + model parameters (mean/std,stride/num feats (last conv)/colorspace format)
    local net = torch.load('../data/pretrained_models/model_'..string.lower(opt.netType)..'.t7')
    net:cuda():evaluate()
    local features = net
    features:remove(features:size())
    features:remove(features:size())
    features:remove(features:size())
    

    local input = torch.randn(1,3,224,224):cuda()
    utils.testSurgery(input, utils.model.DisableFeatureBackprop, features, 5)
    utils.testSurgery(input, inn.utils.foldBatchNorm, features:findModules'nn.NoBackprop'[1])
    utils.testSurgery(input, inn.utils.BNtoFixed, features, true)
    utils.testSurgery(input, inn.utils.BNtoFixed, net, true)


    -- setup classifier
    local classifier = nn.Sequential()
        :add(nn.Linear(nfeats*7*7, 4096))
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
        :add(inn.ROIPooling(7,7,1/32))
        :add(nn.View(-1):setNumInputDims(3))
        :add(utils.makeDataParallel(classifier, opt.nGPU))
        :add(utils.model.CreateClassifierBBoxRegressor(4096, opt.nClasses, opt.has_bbox_regressor))
    
    
    local model_parameters = {
        colourspace = 'rgb',
        meanstd = {mean = {0.485,0.456,0.406}, std = {0.229,0.224, 0.225}},
        pixel_scale = 1,
        stride = 32
    }
    
    return model, model_parameters
end

return CreateModel
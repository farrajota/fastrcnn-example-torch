--[[
    Download ZeilerNet pretrained model on imagenet.

    source: https://github.com/fmassa/object-detection.torch
]]

require 'paths'
require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'inn'
local matio = require 'matio'

------------------------------------------------------------------------------------------------------------

local function createZeiler(model_weights_filepath, backend)
    local maxpooling
    local spatialconv
    local relu

    if backend == 'nn' then
        maxpooling = nn.SpatialMaxPooling
        spatialconv = nn.SpatialConvolution
        relu = nn.ReLU
    elseif backend == 'cudnn' then
        maxpooling = cudnn.SpatialMaxPooling
        spatialconv = cudnn.SpatialConvolution
        relu = cudnn.ReLU
    end

    local features = nn.Sequential()

    local fS =     {96, 256, 384, 384, 256}
    fS[0] = 3
    local ks =     {7,5,3,3,3}
    local stride = {2,2,1,1,1}
    local pad =    {1,0,1,1,1}
    local lrn =    {true,true,false,false,false}
    local pool =   {true,true,false,false,false}

    for i=1,#fS do
      features:add(spatialconv(fS[i-1],fS[i],ks[i],ks[i],stride[i],stride[i],pad[i],pad[i]))
      features:add(relu(true))
      if lrn[i] then
        features:add(inn.SpatialSameResponseNormalization(3,0.00005,0.75))
      end
      if pool[i] then
        features:add(maxpooling(3,3,2,2):ceil())
      end
    end

    local classifier = nn.Sequential()
    local fS = {4096,4096,1000}
    fS[0] = 256*50

    for i=1,#fS do
      classifier:add(nn.Linear(fS[i-1],fS[i]))
      if i < #fS then
        classifier:add(relu(true))
        classifier:add(nn.Dropout(0.5,true))
      end
    end

    local modelpath = model_weights_filepath
    assert(paths.filep(modelpath), 'Parameters file not found: '..modelpath)

    local mat = matio.load(modelpath)
    local idx = 1
    for i=1,features:size() do
      if torch.typename(features:get(i))=='nn.SpatialConvolutionMM' or
        torch.typename(features:get(i))=='nn.SpatialConvolution' or
        torch.typename(features:get(i))=='cudnn.SpatialConvolution' then
        features:get(i).weight:copy(mat['conv'..idx..'_w']:transpose(1,4):transpose(2,3))
        features:get(i).bias:copy(mat['conv'..idx..'_b'])
        idx = idx + 1
      end
    end

    local idx = 6

    for i=1,classifier:size() do
      if torch.typename(classifier:get(i))=='nn.Linear' then
        classifier:get(i).weight:copy(mat['fc'..idx..'_w']:transpose(1,2))
        classifier:get(i).bias:copy(mat['fc'..idx..'_b'])
        idx = idx + 1
      end
    end

    local model = nn.Sequential()
    model:add(features)
    model:add(inn.SpatialPyramidPooling({{1,1},{2,2},{3,3},{6,6}}))
    model:add(classifier)
    return model
end

------------------------------------------------------------------------------------------------------------


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 Fast-RCNN download pretrained models.')
cmd:text()
cmd:text(' ---------- General options ------------------------------------')
cmd:text()
cmd:option('-save_path',   './data', 'Experiment ID')
cmd:text()

local opt = cmd:parse(arg or {})
local savepath = opt.save_path

-- create directory if needed
if not paths.dirp(savepath) then
    print('creating directory: ' .. savepath)
    os.execute('mkdir -p ' .. savepath)
end

print('==> Downloading ZeilerNet model... ')

local url = 'https://www.dropbox.com/s/ge3xxe026udj8js/Zeiler_imagenet_weights.mat?dl=0'

-- file names
local filename_weights = paths.concat(savepath, 'Zeiler_imagenet_weights.mat')

-- download file
-- url1
if not paths.filep(filename_weights) then
  local command = ('wget -O %s %s'):format(filename_weights, url)
  os.execute(command)
end

-- load network
local model = createZeiler(filename_weights, 'cudnn')

-- model's parameters
local params = {}
params.mean =  {128/255,128/255,128/255}
params.pixel_scale = 255.0
params.colourspace = 'rgb'
params.num_feats = 256
params.stride = 16 --pixels

-- save to memory
torch.save(paths.concat(savepath, 'model_zeilernet.t7'), model)
torch.save(paths.concat(savepath, 'parameters_zeilernet.t7'), params)

collectgarbage()

print('Done.')
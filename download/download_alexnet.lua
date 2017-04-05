--[[
    Download AlexNet pretrained model on imagenet.

    source: https://github.com/mahyarnajibi/fast-rcnn-torch
]]


require 'paths'
require 'torch'


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

print('==> Downloading Alexnet model... ')

local url = 'http://www.umiacs.umd.edu/~najibi/data/imgnet_models.tar.gz'

-- file name
local filename_model = paths.concat(savepath, 'Alexnet_weights_imagenet.tar.gz')

-- download file
-- url1
if not paths.filep(filename_model) then
  local command = ('wget -O %s %s'):format(filename_model, url)
  os.execute(command)
end

-- extract file + rename file
local command = ('tar -xvf %s -C %s'):format(filename_model, savepath)
os.execute(command)
local command = ('mv %s %s'):format(paths.concat(savepath, 'imgnet_alexnet.t7'), paths.concat(savepath, 'model_alexnet.t7'))
os.execute(command)

-- model's parameters
local params = {}
params.mean = {102.9801,115.9465,122.7717}
params.pixel_scale = 255.0
params.colourspace = 'bgr'
params.num_feats = 256
params.stride = 16 --pixels

-- save to memory
torch.save(paths.concat(savepath, 'parameters_alexnet.t7'), params)

collectgarbage()

print('Done.')
--[[
    Download GoogleNet pretrained model on imagenet.

    source: https://github.com/facebook/fb.resnet.torch/tree/master/pretrained
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

print('==> Downloading googlenet inception v3 models...')

local url1 = 'https://www.dropbox.com/s/tv4y5445hvsomla/inceptionv3_cudnn.t7?dl=0c' --cudnn
local url2 = 'https://www.dropbox.com/s/tv4y5445hvsomla/inceptionv3_cudnn.t7?dl=0'  --cunn

-- file names
local filename_model1 = paths.concat(savepath, 'model_googlenet_inceptionv3_cudnn.t7')
local filename_model2 = paths.concat(savepath, 'model_googlenet_inceptionv3_cunn.t7')

-- download file
-- url1
print(filename_model1)
if not paths.filep(filename_model1) then
  local command = ('wget -O %s %s'):format(filename_model1, url1)
  os.execute(command)
end
-- url2
print(filename_model2)
if not paths.filep(filename_model2) then
  local command = ('wget -O %s %s'):format(filename_model2, url2)
  os.execute(command)
end


-- model's parameters
local opt = {}
opt.mean = {128,128,128}
opt.std = {0.0078125, 0.0078125, 0.0078125}
opt.pixel_scale = 255.0
opt.colourspace = 'rgb'
opt.num_feats = 2048
opt.stride = 37.375 --pixels

-- save to memory
torch.save(paths.concat(savepath, 'parameters_googlenet_inceptionv3_cudnn.t7'), opt)
torch.save(paths.concat(savepath, 'parameters_googlenet_inceptionv3_cunn.t7'), opt)

collectgarbage()

print('Done.')
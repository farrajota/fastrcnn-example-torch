--[[
    Download models and convert/store them to torch7 file format.
]]

require 'nn'
require 'cunn'
require 'cudnn'
require 'inn'
require 'loadcaffe'
local matio = require 'matio'


local function download_alexnet(savepath, root_folder)
--repo: https://github.com/mahyarnajibi/fast-rcnn-torch

  print('Downloading Alexnet... ')

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
  local opt = {}
  opt.mean = {102.9801,115.9465,122.7717}
  opt.pixel_scale = 255.0
  opt.colourspace = 'bgr'
  opt.num_feats = 256
  opt.stride = 16 --pixels
  
  -- save to memory
  torch.save(paths.concat(savepath, 'parameters_alexnet.t7'), opt)
  
  -- make a symlink if the save folder is different than the pretrained model's root folder (default folder).
  if savepath ~= root_folder then
    print('Making a symbolic link to: ' .. root_folder)
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'model_alexnet.t7'), paths.concat(root_folder, 'model_alexnet.t7')))
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'parameters_alexnet.t7'), paths.concat(root_folder, 'parameters_alexnet.t7')))
  end
  
  collectgarbage()
  print('Done.')
end

----------------------------------------------

local function download_zeiler(savepath, root_folder)
--repo: https://github.com/fmassa/object-detection.torch

  -----------------------------------
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
      assert(paths.filep(modelpath),
        'Parameters file not found: '..modelpath)
      
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
  -----------------------------------
  
  print('Downloading ZeilerNet... ')  
  
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
  local opt = {}
  opt.mean =  {128/255,128/255,128/255}
  opt.pixel_scale = 255.0
  opt.colourspace = 'rgb'
  opt.num_feats = 256
  opt.stride = 16 --pixels
  
  -- save to memory
  torch.save(paths.concat(savepath, 'model_zeilernet.t7'), model)
  torch.save(paths.concat(savepath, 'parameters_zeilernet.t7'), opt)
  
  -- make a symlink if the save folder is different than the pretrained model's root folder (default folder).
  if savepath ~= root_folder then
    print('Making a symbolic link to: ' .. root_folder)
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'model_zeilernet.t7'), paths.concat(root_folder, 'model_zeilernet.t7')))
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'parameters_zeilernet.t7'), paths.concat(root_folder, 'parameters_zeilernet.t7')))
  end
  
  collectgarbage()
  
  print('Done.')
end

----------------------------------------------

local function download_vgg16_vgg19(savepath, root_folder)
--repo (vgg-16): https://gist.github.com/ksimonyan/211839e770f7b538e2d8
--repo (vgg-19): https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
  
  print('Downloading VGG 16 and 19...')
  
  local url1 = 'http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'
  local url2 = 'https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/c3ba00e272d9f48594acef1f67e5fd12aff7a806/VGG_ILSVRC_16_layers_deploy.prototxt'
  local url3 = 'http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel'
  local url4 = 'https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt'
  
  -- file names
  local vgg_16_filename_model = paths.concat(savepath, 'VGG_ILSVRC_16_layers.caffemodel')
  local vgg_16_filename_proto = paths.concat(savepath, 'VGG_ILSVRC_16_layers_deploy.prototxt')
  local vgg_19_filename_model = paths.concat(savepath, 'VGG_ILSVRC_19_layers.caffemodel')
  local vgg_19_filename_proto = paths.concat(savepath, 'VGG_ILSVRC_19_layers_deploy.prototxt')
    
  -- download file
  -- url1
  if not paths.filep(vgg_16_filename_model) then
    local command = ('wget -O %s %s'):format(vgg_16_filename_model, url1)
    os.execute(command)
  end
  -- url2
  if not paths.filep(vgg_16_filename_proto) then
    local command = ('wget -O %s %s'):format(vgg_16_filename_proto, url2)
    os.execute(command)
  end
  -- url3
  if not paths.filep(vgg_19_filename_model) then
    local command = ('wget -O %s %s'):format(vgg_19_filename_model, url3)
    os.execute(command)
  end
  -- url4
  if not paths.filep(vgg_19_filename_proto) then
    local command = ('wget -O %s %s'):format(vgg_19_filename_proto, url4)
    os.execute(command)
  end
  
  -- load network
  local model_vgg16 = loadcaffe.load(vgg_16_filename_proto, vgg_16_filename_model, 'cudnn')
  local model_vgg19 = loadcaffe.load(vgg_19_filename_proto, vgg_19_filename_model, 'cudnn')
  
  -- model's parameters
  local opt = {}
  opt.mean = {103.939, 116.779, 123.68}
  opt.pixel_scale = 255.0
  opt.colourspace = 'bgr'
  opt.num_feats = 512
  opt.stride = 16 --pixels

  -- save to memory
  torch.save(paths.concat(savepath, 'model_vgg16.t7'), model_vgg16)
  torch.save(paths.concat(savepath, 'model_vgg19.t7'), model_vgg19)
  torch.save(paths.concat(savepath, 'parameters_vgg16.t7'), opt)
  torch.save(paths.concat(savepath, 'parameters_vgg19.t7'), opt)
  
  -- make a symlink if the save folder is different than the pretrained model's root folder (default folder).
  if savepath ~= root_folder then
    print('Making a symbolic link to: ' .. root_folder)
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'model_vgg16.t7'), paths.concat(root_folder, 'model_vgg16.t7')))
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'parameters_vgg16.t7'), paths.concat(root_folder, 'parameters_vgg16.t7')))
    
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'model_vgg19.t7'), paths.concat(root_folder, 'model_vgg19.t7')))
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'parameters_vgg19.t7'), paths.concat(root_folder, 'parameters_vgg19.t7')))
  end
  
  collectgarbage()
  
  print('Done.')
end

----------------------------------------------

local function download_resnet(savepath, root_folder)
--repo: https://github.com/facebook/fb.resnet.torch/tree/master/pretrained
  
  print('Downloading resnet-18, resnet-32, resnet-50, resnet-101, resnet-152, resnet-200...')
  
  local url1 = 'https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7'
  local url2 = 'https://d2j0dndfm35trm.cloudfront.net/resnet-34.t7'
  local url3 = 'https://d2j0dndfm35trm.cloudfront.net/resnet-50.t7'
  local url4 = 'https://d2j0dndfm35trm.cloudfront.net/resnet-101.t7'
  local url5 = 'https://d2j0dndfm35trm.cloudfront.net/resnet-152.t7'
  local url6 = 'https://d2j0dndfm35trm.cloudfront.net/resnet-200.t7'
  
  -- file names
  local filename_model18 = paths.concat(savepath, 'model_resnet-18.t7')
  local filename_model32 = paths.concat(savepath, 'model_resnet-32.t7')
  local filename_model50 = paths.concat(savepath, 'model_resnet-50.t7')
  local filename_model101 = paths.concat(savepath, 'model_resnet-101.t7')
  local filename_model152 = paths.concat(savepath, 'model_resnet-152.t7')
  local filename_model200 = paths.concat(savepath, 'model_resnet-200.t7')
  
  -- download file
  -- url1
  if not paths.filep(filename_model18) then
    local command = ('wget -O %s %s'):format(filename_model18, url1)
    os.execute(command)
  end
  -- url2
  if not paths.filep(filename_model32) then
    local command = ('wget -O %s %s'):format(filename_model32, url2)
    os.execute(command)
  end
  -- url3
  if not paths.filep(filename_model50) then
    local command = ('wget -O %s %s'):format(filename_model50, url3)
    os.execute(command)
  end
  -- url4
  if not paths.filep(filename_model101) then
    local command = ('wget -O %s %s'):format(filename_model101, url4)
    os.execute(command)
  end
  -- url5
  if not paths.filep(filename_model152) then
    local command = ('wget -O %s %s'):format(filename_model152, url5)
    os.execute(command)
  end
  -- url6
  if not paths.filep(filename_model200) then
    local command = ('wget -O %s %s'):format(filename_model200, url6)
    os.execute(command)
  end

  -- model's parameters
  local opt = {}
  opt.mean = {0.485, 0.456, 0.406}
  opt.std = {0.229, 0.224, 0.225}
  opt.pixel_scale = 1.0
  opt.colourspace = 'rgb'
  opt.num_feats = 512
  opt.stride = 32 --pixels
  
  -- save to memory
  torch.save(paths.concat(savepath, 'parameters_resnet-18.t7'), opt)
  torch.save(paths.concat(savepath, 'parameters_resnet-32.t7'), opt)
  torch.save(paths.concat(savepath, 'parameters_resnet-50.t7'), opt)
  torch.save(paths.concat(savepath, 'parameters_resnet-101.t7'), opt)
  torch.save(paths.concat(savepath, 'parameters_resnet-152.t7'), opt)
  torch.save(paths.concat(savepath, 'parameters_resnet-200.t7'), opt)
  
  -- make a symlink if the save folder is different than the pretrained model's root folder (default folder).
  if savepath ~= root_folder then
    print('Making a symbolic link to: ' .. root_folder)
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'model_resnet-18.t7'), paths.concat(root_folder, 'model_resnet-18.t7')))
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'model_resnet-32.t7'), paths.concat(root_folder, 'model_resnet-32.t7')))
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'model_resnet-50.t7'), paths.concat(root_folder, 'model_resnet-50.t7')))
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'model_resnet-101.t7'), paths.concat(root_folder, 'model_resnet-101.t7')))
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'model_resnet-152.t7'), paths.concat(root_folder, 'model_resnet-152.t7')))
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'model_resnet-200.t7'), paths.concat(root_folder, 'model_resnet-200.t7')))
    
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'parameters_resnet-18.t7'), paths.concat(root_folder, 'parameters_resnet-18.t7')))
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'parameters_resnet-32.t7'), paths.concat(root_folder, 'parameters_resnet-32.t7')))
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'parameters_resnet-50.t7'), paths.concat(root_folder, 'parameters_resnet-50.t7')))
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'parameters_resnet-101.t7'), paths.concat(root_folder, 'parameters_resnet-101.t7')))
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'parameters_resnet-152.t7'), paths.concat(root_folder, 'parameters_resnet-152.t7')))
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'parameters_resnet-200.t7'), paths.concat(root_folder, 'parameters_resnet-200.t7')))
  end
  
  collectgarbage()
  
  print('Done.')
end

----------------------------------------------

local function download_googlenet(savepath, root_folder)
--repo: https://github.com/facebook/fb.resnet.torch/tree/master/pretrained

  print('Downloading googlenet inception v3...')

  local url1 = 'https://www.dropbox.com/s/tv4y5445hvsomla/inceptionv3_cudnn.t7?dl=0c' --cudnn
  local url2 = 'https://www.dropbox.com/s/tv4y5445hvsomla/inceptionv3_cudnn.t7?dl=0'  --cunn
  
  -- file names
  local filename_model1 = paths.concat(savepath, 'model_googlenet_inceptionv3_cudnn.t7')
  local filename_model2 = paths.concat(savepath, 'model_googlenet_inceptionv3_cunn.t7')

  -- download file
  -- url1
  print(filename_model1)
  print(filename_model2)
  if not paths.filep(filename_model1) then
    local command = ('wget -O %s %s'):format(filename_model1, url1)
    os.execute(command)
  end
  -- url2
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
  
  -- make a symlink if the save folder is different than the pretrained model's root folder (default folder).
  if savepath ~= root_folder then
    print('Making a symbolic link to: ' .. root_folder)
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'model_googlenet_inceptionv3_cudnn.t7'), paths.concat(root_folder, 'model_googlenet_inceptionv3_cudnn.t7')))
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'parameters_googlenet_inceptionv3_cudnn.t7'), paths.concat(root_folder, 'parameters_googlenet_inceptionv3_cudnn.t7')))
    
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'model_googlenet_inceptionv3_cunn.t7'), paths.concat(root_folder, 'model_googlenet_inceptionv3_cunn.t7')))
    os.execute(('ln -s %s %s'):format(paths.concat(savepath, 'parameters_googlenet_inceptionv3_cunn.t7'), paths.concat(root_folder, 'parameters_googlenet_inceptionv3_cunn.t7')))
  end
  
  collectgarbage()
  
  print('Done.')
end

----------------------------------------------

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 pretrained models download script for FAST-RCNN.')
cmd:text()
cmd:text('Options:')
-- data sampling
cmd:option('-save_dir', '../data/pretrained_models', 'Download models to this folder.')
cmd:option('-download_model', 'all',  'Specify which model to download.')

-- parse options
local opt = cmd:parse(arg or {})

-- root folder to store pretrained models (or links)
local root_folder = '../data/pretrained_models'

-- create directory if needed
print('creating directory: ' .. opt.save_dir)
os.execute('mkdir -p ' .. opt.save_dir)
os.execute('mkdir -p ' .. root_folder)

-- download models
print('==> Downloading imagenet models to: ' .. opt.save_dir)
if opt.download_model == 'all' or opt.download_model == 'alexnet'     then download_alexnet(opt.save_dir, root_folder)     end
if opt.download_model == 'all' or opt.download_model == 'zeiler'      then download_zeiler(opt.save_dir, root_folder)      end
if opt.download_model == 'all' or opt.download_model == 'vgg'         then download_vgg16_vgg19(opt.save_dir, root_folder) end
if opt.download_model == 'all' or opt.download_model == 'resnet'      then download_resnet(opt.save_dir, root_folder)      end
if opt.download_model == 'all' or opt.download_model == 'googlenet'   then download_googlenet(opt.save_dir, root_folder)   end

print('==> Imagenet models download complete.')
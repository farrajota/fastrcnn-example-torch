--[[
    Train a Fast-RCNN detector network using the Pascal VOC 2007 dataset.
]]


require 'paths'
require 'torch'
local fastrcnn = require 'fastrcnn'
--local fastrcnn = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/init.lua')

torch.setdefaulttensortype('torch.FloatTensor')


--------------------------------------------------------------------------------
-- Load options
--------------------------------------------------------------------------------

print('==> (1/5) Load options')
local opts = require 'options'
local opt = opts.parse(arg, 'PascalVOC2007')


--------------------------------------------------------------------------------
-- Load dataset data loader
--------------------------------------------------------------------------------
-- The fastrcnn.train() function receives a table with loading functions to fetch
-- the necessary data from a data structure. This way it is easy to use other
-- datasets with the fastrcnn package.

print('==> (2/5) Load dataset data loader')
local data_loader = require 'pascal_voc_2007.data'
local loader = data_loader('train')


--------------------------------------------------------------------------------
-- Load regions-of-interest (RoIs)
--------------------------------------------------------------------------------

print('==> (3/5) Load roi proposals data')
local loadRoiDataFn = fastrcnn.utils.load.matlab.single_file
local rois = {
    train = loadRoiDataFn(paths.concat('data','proposals', 'selective_search_data', 'voc_2007_trainval.mat')),
    test =  loadRoiDataFn(paths.concat('data','proposals', 'selective_search_data', 'voc_2007_test.mat'))
}


--------------------------------------------------------------------------------
-- Setup model
--------------------------------------------------------------------------------

local model, model_parameters
if opt.loadModel == '' then
    print('==> (4/5) Setup model:')
    local model = require 'models'
    model, model_parameters = model(opt.netType, opt.nGPU, 20, opt.has_bbox_regressor)
else
    print('==> (4/5) Load model from file: ')
    _, model_parameters = model(opt.netType, opt.nGPU, 20, opt.has_bbox_regressor)
    model = torch.load(opt.loadModel)
end


--------------------------------------------------------------------------------
-- Train a  Fast R-CNN detector
--------------------------------------------------------------------------------

print('==> (5/5) Train Fast-RCNN model')
fastrcnn.train(loader, rois, model, model_parameters, opt)

print('Script complete.')
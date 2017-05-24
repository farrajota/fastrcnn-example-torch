--[[
    Train a Fast-RCNN detector network using the Pascal VOC 2007/MSCOCO dataset.
]]


require 'paths'
require 'torch'
local fastrcnn = require 'fastrcnn'

torch.setdefaulttensortype('torch.FloatTensor')
paths.dofile('projectdir.lua')


--------------------------------------------------------------------------------
-- Load options
--------------------------------------------------------------------------------

print('==> (1/5) Load options')
local opts = paths.dofile('options.lua')
local opt = opts.parse(arg)


--------------------------------------------------------------------------------
-- Load dataset data loader
--------------------------------------------------------------------------------
-- The fastrcnn.train() function receives a table with loading functions to fetch
-- the necessary data from a data structure. This way it is easy to use other
-- datasets with the fastrcnn package.

print('==> (2/5) Load dataset data loader')
local data_loader = paths.dofile('data.lua')
local data_gen = data_loader(opt.dataset, 'train')


--------------------------------------------------------------------------------
-- Load regions-of-interest (RoIs)
--------------------------------------------------------------------------------

print('==> (3/5) Load roi proposals data')
local rois_loader = paths.dofile('rois.lua')
local rois = rois_loader(opt.dataset, 'train')


--------------------------------------------------------------------------------
-- Setup model
--------------------------------------------------------------------------------

local model, model_parameters
if opt.loadModel == '' then
    print('==> (4/5) Setup model:')
    local nclasses = (opt.dataset=='mscoco' and 80) or 20
    local load_model = paths.dofile('models/init.lua')
    model, model_parameters = load_model(opt.netType, opt.nGPU, nclasses)
else
    print('==> (4/5) Load model from file: ')
    local model_data = torch.load(opt.load)
    model, model_parameters = model_data.model, model_data.params
end


--------------------------------------------------------------------------------
-- Train a  Fast R-CNN detector
--------------------------------------------------------------------------------

print('==> (5/5) Train Fast-RCNN model')
fastrcnn.train(data_gen, rois, model, model_parameters, opt)

print('Script complete.')
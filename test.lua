--[[
    Test a Fast-RCNN detector network using the Pascal VOC 2007 dataset.
]]


require 'paths'
require 'torch'
--local fastrcnn = require 'fastrcnn'
local fastrcnn = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/init.lua')

torch.setdefaulttensortype('torch.FloatTensor')


--------------------------------------------------------------------------------
-- Load options
--------------------------------------------------------------------------------

print('==> (1/5) Load options')
local opts = paths.dofile('options.lua')
--local opts = require 'options'
local opt = opts.parse(arg)


--------------------------------------------------------------------------------
-- Load dataset data loader
--------------------------------------------------------------------------------
-- The fastrcnn.train() function receives a table with loading functions to fetch
-- the necessary data from a data structure. This way it is easy to use other
-- datasets with the fastrcnn package.

print('==> (2/5) Load dataset data loader')
--local data_loader = require 'pascal_voc_2007.data'
local data_loader = paths.dofile('data.lua')
local data_gen = data_loader(opt.dataset, 'test')


--------------------------------------------------------------------------------
-- Load regions-of-interest (RoIs)
--------------------------------------------------------------------------------

print('==> (3/5) Load roi proposals data')
local rois_loader = paths.dofile('rois.lua')
local rois = rois_loader(opt.dataset, 'test')


--------------------------------------------------------------------------------
-- Setup model
--------------------------------------------------------------------------------

print('==> (4/5) Load model: ' .. opt.load)
local model, model_parameters = unpack(torch.load(opt.load))


--------------------------------------------------------------------------------
-- Test detector mAP
--------------------------------------------------------------------------------

print('==> (5/5) Test Fast-RCNN model')
fastrcnn.test(data_gen, rois, model, model_parameters, opt)

print('Script complete.')
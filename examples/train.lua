--[[
    Train a Fast-RCNN detector network using the Pascal VOC 2007 dataset.
]]


require 'paths'
require 'torch'
paths.dofile('../code/projectdir.lua')
local fastrcnn = paths.dofile(paths.concat(projectDir,'init.lua')) -- load fastrcnn package
--local fastrcnn = require 'fastrcnn' -- load fastrcnn package

torch.setdefaulttensortype('torch.FloatTensor')

--------------------------------------------------------------------------------
-- Load options
--------------------------------------------------------------------------------

print('==> (1/5) Load options')
local opts = paths.dofile(paths.concat(projectDir, 'code', 'options.lua'))
local opt = opts.parse(arg)


--------------------------------------------------------------------------------
-- Load dataset
--------------------------------------------------------------------------------

print('==> (2/5) Load dataset')
local dbclt = require 'dbcollection'
local dataset, train_rois_file, test_rois_file
local loadRoiDataFn = fastrcnn.utils.data.loadmatlab.loadSingleFile
if opt.dataset == 'pascalvoc2007' then
    dataset = dbclt.get{name = 'pascalvoc2007', verbose = true, category = 'full', task = 'detection', verbose = opt.verbose}
    train_rois_file = paths.concat(projectDir, 'data','proposals', 'selective_search_data', 'voc_2007_trainval.mat')
    test_rois_file = paths.concat(projectDir, 'data','proposals', 'selective_search_data', 'voc_2007_test.mat')
elseif opt.dataset == 'pascalvoc2012' then
    dataset = dbclt.get{name = 'pascalvoc2012', verbose = true, category = 'full', task = 'detection'}
    train_rois_file = paths.concat(projectDir, 'data','proposals','selective_search_data','voc_2012_trainval.mat')
    test_rois_file = paths.concat(projectDir, 'data','proposals','selective_search_data','voc_2012_test.mat')
elseif opt.dataset == 'mscoco' then
    dataset = dbclt.get{name = 'mscoco', verbose = true, category = 'full', task = 'detection'}
    train_rois_file = paths.concat(projectDir, 'data','proposals','MCG')
    test_rois_file = paths.concat(projectDir, 'data','proposals','MCG')
    loadRoiDataFn = fastrcnn.utils.data.loadmatlab.loadMultiFiles
end


--------------------------------------------------------------------------------
-- Load regions-of-interest (RoIs)
--------------------------------------------------------------------------------

print('==> (3/5) Load roi proposals data')
local rois = {
    train = loadRoiDataFn(train_rois_file),
    test =  loadRoiDataFn(test_rois_file)
}


--------------------------------------------------------------------------------
-- Create model
--------------------------------------------------------------------------------

print('==> (4/5) Load model:')
print('Load featuresNet')
local featuresNet, model_parameters = fastrcnn.utils.model.setup.LoadFeatures(opt.netType, paths.concat(projectDir, 'data/pretrained_models'))
print('Setup network')
local model = fastrcnn.utils.model.setup.CreateModel(featuresNet, model_parameters, #dataset.data.train.classLabel, opt)


--------------------------------------------------------------------------------
-- Train detector
--------------------------------------------------------------------------------

print('==> (5/5) Train Fast-RCNN model')
fastrcnn.train(dataset, rois, model, model_parameters)

print('Script complete.')
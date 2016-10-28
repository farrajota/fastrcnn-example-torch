--[[
    Test a Fast-RCNN detector network using the Pascal VOC 2007 dataset.
]]


require 'paths'
require 'torch'
--local fastrcnn = require 'fastrcnn' -- load fastrcnn package
local fastrcnn = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/init.lua')
paths.dofile('projectdir.lua')

torch.setdefaulttensortype('torch.FloatTensor')

--------------------------------------------------------------------------------
-- Load options
--------------------------------------------------------------------------------

print('==> (1/5) Load options')
local opt = fastrcnn.options.parse(arg)


--------------------------------------------------------------------------------
-- Load dataset
--------------------------------------------------------------------------------

print('==> (2/5) Load dataset')
local dbclt = require 'dbcollection'
local dataset, train_rois_file, test_rois_file
local loadRoiDataFn = fastrcnn.utils.loadmatlab.loadSingleFile
if opt.dataset == 'pascalvoc2007' then
    dataset = dbclt.get{name = 'pascalvoc2007', verbose = true, category = 'no_difficult', task = 'detection', verbose = opt.verbose}
    test_rois_file = paths.concat(projectDir, 'data','proposals', 'selective_search_data', 'voc_2007_test.mat')
elseif opt.dataset == 'pascalvoc2012' then
    dataset = dbclt.get{name = 'pascalvoc2012', verbose = true, category = 'no_difficult', task = 'detection'}
    test_rois_file = paths.concat(projectDir, 'data','proposals','selective_search_data','voc_2012_test.mat')
elseif opt.dataset == 'mscoco' then
    dataset = dbclt.get{name = 'mscoco', verbose = true, category = 'full', task = 'detection'}
    test_rois_file = paths.concat(projectDir, 'data','proposals','MCG')
    loadRoiDataFn = fastrcnn.utils.data.loadmatlab.loadMultiFiles
end


--------------------------------------------------------------------------------
-- Load regions-of-interest (RoIs)
--------------------------------------------------------------------------------

print('==> (3/5) Load roi proposals data')
local rois = {
    test =  loadRoiDataFn(test_rois_file)
}


--------------------------------------------------------------------------------
-- Load/setup model
--------------------------------------------------------------------------------

print('==> (4/5) Load model: ' .. paths.concat(opt.save, opt.load))
local model = torch.load(paths.concat(opt.save, opt.load))

--[[

local filename = '/home/mf/multipathnet/logs/fastrcnn_voc2007_3136927341/model_final.t7'
local filename = '/home/mf/multipathnet/logs/fastrcnn_voc2007_717913989/model_final.t7'
local filename = '/home/mf/Toolkits/Codigo/git/fastrcnn-example/data/exp/pascalvoc2007/alexnet_sgd3/model_final.t7'
local model = torch.load(filename)
print('==> (4/5) Load model: ' .. filename)
--]]

local model_parameters = torch.load(paths.concat(opt.save, 'model_parameters.t7'))
print(model)

--------------------------------------------------------------------------------
-- Test detector mAP
--------------------------------------------------------------------------------

print('==> (5/5) Test Fast-RCNN model')
opt.model_param = model_parameters
fastrcnn.test(dataset, rois, model, model_parameters, opt)

print('Script complete.')

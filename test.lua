--[[
    Test a Fast-RCNN detector network using the Pascal VOC 2007 dataset.
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
if opt.frcnn_test_mode == 'voc' then
    fastrcnn.test(data_gen, rois, model, model_parameters, opt)
else
    local annotation_file
    if opt.dataset == 'pascal_voc_2007' then
        annotation_file = projectDir ..  '/data/coco_eval_annots/pascal_test2007.json'
    elseif opt.dataset == 'pascal_voc_2012' then
        annotation_file = projectDir ..  '/data/coco_eval_annots/pascal_val2012.json'
    elseif opt.dataset == 'pascal_voc_2007_2012' then
        annotation_file = projectDir ..  '/data/coco_eval_annots/pascal_test2007.json'
    elseif opt.dataset == 'coco' then
        annotation_file = projectDir ..  '/data/coco_eval_annots/instances_val2014.json'
    else
        error(('Invalid dataset: %s. Available datasets: pascal_voc_2007, pascal_voc_2012, pascal_voc_2007_2012 or coco'):format(name))
    end

    fastrcnn.test(data_gen, rois, model, model_parameters, opt, annotation_file)
end

print('Script complete.')
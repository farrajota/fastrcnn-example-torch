--[[
    Fast-RCNN demo. Select a random image from the dataset and proceeds to display all detected objects over a certain threshold.
]]


require 'paths'
require 'torch'
local ffi = require 'ffi'
--local fastrcnn = require 'fastrcnn' -- load fastrcnn package
paths.dofile('../code/projectdir.lua')
local fastrcnn = paths.dofile(paths.concat(projectDir,'init.lua')) -- load fastrcnn package

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
    test_rois_file = paths.concat(projectDir, 'data','proposals', 'selective_search_data', 'voc_2007_test.mat')
elseif opt.dataset == 'pascalvoc2012' then
    dataset = dbclt.get{name = 'pascalvoc2012', verbose = true, category = 'full', task = 'detection'}
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
local model_parameters = torch.load(paths.concat(opt.save, 'model_parameters.t7'))


--------------------------------------------------------------------------------
-- Load detector class
--------------------------------------------------------------------------------

local imdetector = fastrcnn.Detector(model, opt, model_parameters)


--------------------------------------------------------------------------------
-- Fetch random image + roi proposals
--------------------------------------------------------------------------------

print('==> (4/6) Load test image + proposals boxes')
-- Loading the image
local randIdx = torch.random(1, dataset.data.test.filename:size(1))
local im = image.load(ffi.string(dataset.data.test.filename[randIdx]:data()))
local proposals = rois.test[randIdx]


--------------------------------------------------------------------------------
-- Process detection
--------------------------------------------------------------------------------

-- (5) Process image detection with the FRCNN
print('==> (5/6) Process image detections')
local scores, bboxes = imdetector:detect(im, proposals)


--------------------------------------------------------------------------------
-- Visualize detections
--------------------------------------------------------------------------------

print('==> (6/6) Visualize detections')
local threshold = 0.5 -- score threshold for visualization purposes only
local classes = dataset.data.test.classLabel
fastrcnn.utils.visualize_detections(im, bboxes, scores, threshold, opt.frcnn_test_nms_thresh, classes)

print('Script complete.')



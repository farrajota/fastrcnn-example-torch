--[[
    Fast-RCNN demo. Applies detection on an available standalone test image available in the package.
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

print('==> (1/6) Load options')
local opts = paths.dofile(paths.concat(projectDir, 'code', 'options.lua'))
local opt = opts.parse(arg)


--------------------------------------------------------------------------------
-- Load/setup model
--------------------------------------------------------------------------------

print('==> (2/6) Load model: ' .. paths.concat(opt.save, opt.load))
local model = torch.load(paths.concat(opt.save, opt.load))
local model_parameters = torch.load(paths.concat(opt.save, 'model_parameters.t7'))


--------------------------------------------------------------------------------
-- Setup detector class
--------------------------------------------------------------------------------

print('==> (3/6) Setup detector class')
local imdetector = fastrcnn.Detector(model, opt, model_parameters)


--------------------------------------------------------------------------------
-- Load image + roi proposals
--------------------------------------------------------------------------------

print('==> (4/6) Load test image + proposals boxes')
-- Loading the image
local im = image.load(paths.concat(projectDir,'data/demo/test.jpg'))
local proposals = fastrcnn.utils.data.loadmatlab.loadSingleFile(paths.concat(projectDir,'data/demo/proposals.mat')):float()


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
-- classes from Pascal used for training the model
local classes = {'aeroplane','bicycle','bird','boat','bottle','bus','car',
'cat','chair','cow','diningtable','dog','horse','motorbike',
'person','pottedplant','sheep','sofa','train','tvmonitor'}
-- visualize results
fastrcnn.utils.visualize_detections(im, bboxes, scores, threshold, opt.frcnn_test_nms_thresh, classes)

print('Script complete.')



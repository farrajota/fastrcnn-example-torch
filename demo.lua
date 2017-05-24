--[[
    Fast-RCNN demo.

    Performs object detection on a single test image (random).
]]


require 'paths'
require 'torch'
local fastrcnn = require 'fastrcnn'

torch.setdefaulttensortype('torch.FloatTensor')


--------------------------------------------------------------------------------
-- Load options
--------------------------------------------------------------------------------

print('==> (1/7) Load options')
local opts = paths.dofile('options.lua')
local opt = opts.parse(arg)


--------------------------------------------------------------------------------
-- Load/setup model
--------------------------------------------------------------------------------

print('==> (2/7) Load model: ' .. opt.load)
local model, model_parameters = unpack(torch.load(opt.load))


--------------------------------------------------------------------------------
-- Load regions-of-interest (RoIs)
--------------------------------------------------------------------------------

print('==> (3/7) Load roi proposals data')
local rois_loader = paths.dofile('rois.lua')
local rois = rois_loader(opt.dataset, 'test')


--------------------------------------------------------------------------------
-- Setup detector class
--------------------------------------------------------------------------------

print('==> (4/7) Setup detector class')
local imdetector = fastrcnn.ImageDetector(model, model_parameters, opt) -- single image detector/tester


--------------------------------------------------------------------------------
-- Load image + roi proposals
--------------------------------------------------------------------------------

print('==> (5/7) Load test image + proposals boxes')
local data_loader = paths.dofile('data.lua')
local loader = data_loader(opt.dataset, 'test')()

local randIdx = torch.random(1, loader.test.nfiles)
local im = image.load(loader.test.getFilename(randIdx), 3, 'float')
local proposals = rois.test[randIdx]:float()


--------------------------------------------------------------------------------
-- Process detection
--------------------------------------------------------------------------------

print('==> (6/7) Process image detections')
local scores, bboxes = imdetector:detect(im, proposals)


--------------------------------------------------------------------------------
-- Visualize detections
--------------------------------------------------------------------------------

print('==> (7/7) Visualize detections')
local threshold = 0.5 -- score threshold for visualization purposes only

-- classes from Pascal used for training the model
local classes = loader.test.classLabel

-- visualize results
fastrcnn.visualize_detections(im, bboxes, scores, threshold, opt.frcnn_test_nms_thresh, classes)

print('Script complete.')



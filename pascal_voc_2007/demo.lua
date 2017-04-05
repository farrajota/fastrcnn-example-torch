--[[
    Fast-RCNN demo. Applies detection on an available standalone test image available in the package.
]]


require 'paths'
require 'torch'
local fastrcnn = require 'fastrcnn'
--local fastrcnn = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/init.lua')

torch.setdefaulttensortype('torch.FloatTensor')


--------------------------------------------------------------------------------
-- Load options
--------------------------------------------------------------------------------

print('==> (1/6) Load options')
local opts = require 'options'
local opt = opts.parse(arg, 'PascalVOC2007')


--------------------------------------------------------------------------------
-- Load/setup model
--------------------------------------------------------------------------------

local fname = paths.concat(opt.savedir, opt.load)
print('==> (2/6) Load model: ' .. fname)
local model, model_parameters = unpack(torch.load(fname))


--------------------------------------------------------------------------------
-- Setup detector class
--------------------------------------------------------------------------------

print('==> (3/6) Setup detector class')
local imdetector = fastrcnn.ImageDetector(model, model_parameters, opt) -- single image detector/tester


--------------------------------------------------------------------------------
-- Load image + roi proposals
--------------------------------------------------------------------------------

print('==> (4/6) Load test image + proposals boxes')
local data_loader = require 'pascal_voc_2007.data'
local loader = data_loader('test')

local randIdx = torch.random(1, loader.test.size)
local im = image.load(loader.test.getFilename(randIdx), 3, 'float')
local proposals = rois.test[randIdx]:float()


--------------------------------------------------------------------------------
-- Process detection
--------------------------------------------------------------------------------

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
fastrcnn.visualize_detections(im, bboxes, scores, threshold, opt.frcnn_test_nms_thresh, classes)

print('Script complete.')



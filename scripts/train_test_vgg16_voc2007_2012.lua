--[[
    Train+test a Fast R-CNN network using the VGG16 for feature extraction
    on the Pascal VOC 2007 + 2012 datasets (evaluated on the VOC 2007).
]]


require 'paths'
require 'torch'


--------------------------------------------------------------------------------
-- Train+Test network
--------------------------------------------------------------------------------

local info = {
    -- experiment id
    expID = 'frcnn_vgg16_voc2007_2012',

    -- dataset setup
    dataset = 'pascal_voc_2007_2012',

    -- model setup
    netType = 'vgg16',
    clear_buffers = 'true',

    -- train options
    optMethod = 'sgd',
    nThreads = 4,
    trainIters = 1000,
    snapshot = 10,
    schedule = "{{30,1e-3,5e-4},{10,1e-4,5e-4}}",
    testInter = 'false',
    snapshot = 10,
    nGPU = 1,

    -- FRCNN options
    frcnn_scales = 600,
    frcnn_max_size = 1000,
    frcnn_imgs_per_batch = 2,
    frcnn_rois_per_img = 128,
    frcnn_fg_fraction = 0.25,
    frcnn_bg_fraction = 1.00,
    frcnn_fg_thresh = 0.5,
    frcnn_bg_thresh_hi = 0.5,
    frcnn_bg_thresh_lo = 0.1,
    frcnn_hflip = 0.5,
    frcnn_roi_augment_offset = 0.0,

    -- FRCNN Test options
    frcnn_test_scales = 600,
    frcnn_test_max_size = 1000,
    frcnn_test_nms_thresh = 0.3,
    frcnn_test_mode = 'voc',
    frcnn_test_use_cache = 'false'
}

-- concatenate options fields to a string
local str_args = ''
for k, v in pairs(info) do
    str_args = str_args .. ('-%s %s '):format(k, v)
end

-- display options
print('Input options: ' .. str_args)

-- train network
os.execute(('th train.lua %s'):format(str_args))

-- benchmark network
os.execute(('th test.lua %s'):format(str_args))
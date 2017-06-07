--[[
    Fast-RCNN Torch7 implementation. Options script.
]]

projectDir = projectDir or './'

local options = {}

function options.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text(' ***************************************************************')
    cmd:text(' Torch-7 Fast-RCNN script.')
    cmd:text(' ***************************************************************')
    cmd:text()
    cmd:text(' ---------- General options ------------------------------------')
    cmd:text()
    cmd:option('-expID',      'alexnet_frcnn', 'Experiment ID')
    cmd:option('-expDir',   projectDir .. 'data/exp',  'Experiments directory')
    cmd:option('-dataset',  'pascal_voc_2007', 'Dataset to use for train/test/demo. Options: ' ..
                                               'pascal_voc_2007, pascal_voc_2012, pascal_voc_2007_2012 or coco.')
    cmd:option('-manualSeed',          2, 'Manually set RNG seed')
    cmd:option('-GPU',                 1, 'Default preferred GPU, if set to -1: no GPU')
    cmd:option('-nGPU',                1, 'Number of GPUs to use by default')
    cmd:option('-nThreads',            4, 'Number of data loading threads')
    cmd:option('-verbose',        "true", 'Output messages on screen.')
    cmd:option('-progressbar',    "false", 'Display batch messages using a progress bar if true, else display a more verbose text info.')
    cmd:option('-printConfusion', "false", 'Print confusion matrix into the screen.')
    cmd:text()
    cmd:text(' ---------- Model options --------------------------------------')
    cmd:text()
    cmd:option('-netType',     'alexnet', 'Options: alexnet | vgg16 | vgg19 | resnet-18 | resnet-34 | resnet-50 | ' ..
                                                   'resnet-101 | resnet-152 | resnet-200 | zeiler | googlenetv3.')
    cmd:option('-loadModel',          '', 'Provide the path of a previously trained model')
    cmd:option('-continue',      "false", 'Pick up where an experiment left off')
    cmd:option('-clear_buffers', 'true', 'Empty network\'s buffers (gradInput, etc.) before saving the network to disk (if true).')
    cmd:text()
    cmd:text(' ---------- Hyperparameter options -----------------------------')
    cmd:text()
    cmd:option('-LR',               1e-3, 'Learning rate')
    cmd:option('-LRdecay',           0.0, 'Learning rate decay')
    cmd:option('-momentum',          0.9, 'Momentum')
    cmd:option('-weightDecay',      5e-4, 'Weight decay')
    cmd:option('-optMethod',       'sgd', 'Optimization method: rmsprop | sgd | nag | adadelta | adam')
    cmd:option('-threshold',        .001, 'Threshold (on validation accuracy growth) to cut off training early')
    cmd:text()
    cmd:text(' ---------- Training options -----------------------------------')
    cmd:text()
    cmd:option('-trainIters',     1000, 'Number of train iterations per epoch')
    cmd:option('-epochStart',        1, 'Manual epoch number (useful on restarts)')
    cmd:option('-schedule', "{{30,1e-3,5e-4},{10,1e-4,5e-4}}", 'Optimization schedule. Overrides the previous configs if not empty.')
    cmd:option('-snapshot',          10, 'How often to take a snapshot of the model (0 = never)')
    cmd:option('-testInter',      "true", 'If true, does intermediate testing of the model. Else it only tests the network at the end of the train.')
    cmd:text()
    cmd:text()
    cmd:text(' ===============================================================')
    cmd:text(' ========== ***Fast RCNN options*** ============================')
    cmd:text(' ===============================================================')
    cmd:text()
    cmd:text(' ---------- FRCNN Train options --------------------------------------')
    cmd:text()
    cmd:option('-frcnn_scales',          600, 'Image scales -- the short edge of input image.')
    cmd:option('-frcnn_max_size',       1000, 'Max pixel size of a scaled input image.')
    cmd:option('-frcnn_imgs_per_batch',    2, 'Images per batch.')
    cmd:option('-frcnn_rois_per_img',    128, 'mini-batch size (1 = pure stochastic)')
    cmd:option('-frcnn_fg_fraction',    0.25, 'Fraction of minibatch that is foreground labeled (class > 0).')
    cmd:option('-frcnn_bg_fraction',    1.00, 'Fraction of background samples that has overlap with objects (overlap >= bg_thresh_lo).')
    cmd:option('-frcnn_fg_thresh',       0.5, 'Overlap threshold for a ROI to be considered foreground ' ..
                                              '(if >= fg_thresh).')
    cmd:option('-frcnn_bg_thresh_hi',    0.5, 'Overlap threshold for a ROI to be considered background ' ..
                                              '(class = 0 if overlap in [frcnn_bg_thresh_lo, frcnn_bg_thresh_hi))')
    cmd:option('-frcnn_bg_thresh_lo',    0.1, 'Overlap threshold for a ROI to be considered background ' ..
                                              '(class = 0 if overlap in [frcnn_bg_thresh_lo, frcnn_bg_thresh_hi)).')
    cmd:option('-frcnn_bbox_thresh',     0.5, 'Valid training sample (IoU > bbox_thresh) for bounding box regresion.')
    cmd:text()
    cmd:text(' ---------- FRCNN Test options --------------------------------------')
    cmd:text()
    cmd:option('-frcnn_test_scales',      600, 'Image scales -- the short edge of input image.')
    cmd:option('-frcnn_test_max_size',   1000, 'Max pixel size of a scaled input image.')
    cmd:option('-frcnn_test_max_boxes_split',  2000, 'Split boxes proposals into segments of maximum size \'N\' (helps in out-of-memory situations)')
    cmd:option('-frcnn_test_nms_thresh',  0.3, 'Non-Maximum suppression threshold.')
    cmd:option('-frcnn_test_mode',      "voc", 'mAP testing format: voc or coco.')
    cmd:text()
    cmd:text(' ---------- FRCNN data augment options --------------------------------------')
    cmd:text()
    cmd:option('-frcnn_hflip',         0.5, 'Probability to flip the image horizontally [0,1].')
    cmd:option('-frcnn_roi_augment_offset', 0.0, 'Increase the number of region proposals used for train between a range of coordinates defined by this value [0,1].')
    cmd:text()

    -- parse options
    local opt = cmd:parse(arg or {})

    ---------------------------------------------------------------------------------------------------
    local function ConvertString2Boolean(var) -- converts string to booleans
        if type(var) == 'string' then
            local str = string.lower(var):gsub("%s+", "")
            str = string.gsub(str, "%s+", "")
            if str == 'true' then
                return true
            elseif str == 'false' then
                return false
            else
                error('Cannot convert input to boolean type: ' .. var)
            end
        elseif type(var) == 'boolean' then
            return var
        else
            error('Input variable is not of string/boolean type: ' .. type(var))
        end
    end
    ---------------------------------------------------------------------------------------------------
    local function Str2TableFn(input) -- convert a string into a table
        local json = require 'json'
        -- replace '{' and '}' by '[' and '], respectively
        input = input:gsub("%{","[")
        input = input:gsub("%}","]")
        return json.decode(input)-- use json decode function to convert the string into a table
    end
    ---------------------------------------------------------------------------------------------------


    opt.expDir = paths.concat(opt.expDir, opt.dataset or 'db')
    opt.savedir = paths.concat(opt.expDir, opt.expID)
    opt.load = (opt.loadModel and opt.loadModel ~= '') or paths.concat(opt.savedir, 'model_final.t7')

    -- check if some booleans were inserted as strings. If so, convert the string to boolean type
    opt.continue = ConvertString2Boolean(opt.continue)
    opt.verbose = ConvertString2Boolean(opt.verbose)

    -- check if some booleans were inserted as strings. If so, convert the string to boolean type
    opt.progressbar = ConvertString2Boolean(opt.progressbar)
    opt.printConfusion = ConvertString2Boolean(opt.printConfusion)

    -- convert string to table
    opt.schedule = Str2TableFn(opt.schedule)

    return opt
end

return options
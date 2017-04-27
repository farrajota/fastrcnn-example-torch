--[[
    Load rois for a dataset.
]]


local matio = require 'matio'
local fastrcnn = paths.dofile('/home/mf/Toolkits/Codigo/git/fastrcnn/init.lua')
local loadRoiDataFn = fastrcnn.utils.load.matlab.single_file

------------------------------------------------------------------------------------------------------------

local function select_rois_pascal_2007(mode)
    local str = string.lower(name)
    if str == 'train' then
        return  {
            train = loadRoiDataFn('./data/proposals/selective_search_data/voc_2007_trainval.mat'),
            test = loadRoiDataFn('./data/proposals/selective_search_data/voc_2007_test.mat')
        }
    elseif str == 'test' then
        return  {
            test = loadRoiDataFn('./data/proposals/selective_search_data/voc_2007_test.mat')
        }
    else
        error(('Invalid mode: %s. Available modes: train or test.'):format(mode))
    end
end

------------------------------------------------------------------------------------------------------------

--[[TODO]]
local function select_rois_mscoco(mode)
    local str = string.lower(name)
    if str == 'train' then
        return  {
            train = {},
            test = {}
        }
    elseif str == 'test' then
        return  {
            test = {}
        }
    else
        error(('Invalid mode: %s. Available modes: train or test.'):format(mode))
    end
end

------------------------------------------------------------------------------------------------------------

local function select_rois_dataset(name, mode)
    local str = string.lower(name)
    if str == 'pascal_voc_2007' then
        return select_rois_pascal_2007(mode)
    elseif str == 'mscoco' then
        return select_rois_mscoco(mode)
    else
        error(('Invalid dataset: %s. Available datasets: pascal_voc_2007 or mscoco'):format(name))
    end
end

------------------------------------------------------------------------------------------------------------

local function load_rois(name, mode)
--[[Load rois bboxes of all files into memory]]

    assert(name, 'Undefined dataset name: ' .. name)
    assert(mode == 'train' or mode == 'test', ('Invalid mode: %s. Valid modes: train or test.'):format(mode))

    local save_dir = './data/proposals'

    local proposals_fname = ('%s/%s_%s.t7'):format(save_dir, name, mode)

    -- check if the cache proposals .t7 file exists
    if paths.filep(proposals_fname) then
        return torch.load(proposals_fname)
    else
        local rois = select_rois_dataset(name, mode) -- load roi files
        torch.save(proposals_fname, rois) -- save to .t7 file
        return rois
    end
end

------------------------------------------------------------------------------------------------------------

return load_rois
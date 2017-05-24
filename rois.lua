--[[
    Load rois for a dataset.
]]


local matio = require 'matio'
local utils = require 'fastrcnn.utils'
local loadRoiDataFn = utils.load.matlab.single_file
local loadRoiDataDirFn = utils.load.matlab.load_dir

------------------------------------------------------------------------------------------------------------

local function preprocess_rois_coco()
    local cache_dir = paths.concat(projectDir, 'data', 'cache')
    if not paths.dirp(cache_dir) then
        print('Creating cache dir: ' .. cache_dir)
        os.execute('mkdir -p ' .. cache_dir)
    end

    local train_cache_fname = paths.concat(cache_dir, 'mscoco_proposals_train.t7')
    local test_cache_fname = paths.concat(cache_dir, 'mscoco_proposals_val.t7')
    local tensor_type = 'torch.IntTensor'

    if not paths.filep(train_cache_fname) then
        print('Processing COCO train RoI proposals...')
        local train_rois = loadRoiDataDirFn(paths.concat(projectDir, 'data', 'proposals', 'MCG-COCO-train2014-boxes'), tensor_type)
        print('Save COCO train RoI proposals to cache: ' .. train_cache_fname)
        torch.save(train_cache_fname, train_rois)
    end

    if not paths.filep(test_cache_fname) then
        print('Processing COCO val RoI proposals...')
        local test_rois = loadRoiDataDirFn(paths.concat(projectDir, 'data', 'proposals', 'MCG-COCO-val2014-boxes'), tensor_type)
        print('Save COCO val RoI proposals to cache: ' .. test_cache_fname)
        torch.save(test_cache_fname, test_rois)
    end
    return train_cache_fname, test_cache_fname
end

------------------------------------------------------------------------------------------------------------

local function select_rois_mscoco(mode)
    local train_cache_fname, test_cache_fname = preprocess_rois_coco()
    local str = string.lower(mode)
    if str == 'train' then
        return  {
            train = torch.load(train_cache_fname),
            test = torch.load(test_cache_fname)
        }
    elseif str == 'test' then
        return  {
            test = torch.load(test_cache_fname)
        }
    else
        error(('Invalid mode: %s. Available modes: train or test.'):format(mode))
    end
end

------------------------------------------------------------------------------------------------------------

local function select_rois_pascal_2007(mode)
    local str = string.lower(mode)
    if str == 'train' then
        return  {
            train = loadRoiDataFn(projectDir .. 'data/proposals/selective_search_data/voc_2007_trainval.mat'),
            test = loadRoiDataFn(projectDir .. 'data/proposals/selective_search_data/voc_2007_test.mat')
        }
    elseif str == 'test' then
        return  {
            test = loadRoiDataFn(projectDir .. 'data/proposals/selective_search_data/voc_2007_test.mat')
        }
    else
        error(('Invalid mode: %s. Available modes: train or test.'):format(mode))
    end
end

------------------------------------------------------------------------------------------------------------

local function select_rois_pascal_2012(mode)
    local str = string.lower(mode)
    if str == 'train' then
        return  {
            train = loadRoiDataFn(projectDir .. 'data/proposals/selective_search_data/voc_2012_train.mat'),
            test = loadRoiDataFn(projectDir .. 'data/proposals/selective_search_data/voc_2012_val.mat')
        }
    elseif str == 'test' then
        return  {
            test = loadRoiDataFn(projectDir .. 'data/proposals/selective_search_data/voc_2012_val.mat')
        }
    else
        error(('Invalid mode: %s. Available modes: train or test.'):format(mode))
    end
end

------------------------------------------------------------------------------------------------------------

local function select_rois_pascal_2007_2012(mode)
    local str = string.lower(mode)
    if str == 'train' then
        local train_2007 = loadRoiDataFn(projectDir .. 'data/proposals/selective_search_data/voc_2007_trainval.mat')
        local train_2012 = loadRoiDataFn(projectDir .. 'data/proposals/selective_search_data/voc_2012_trainval.mat')
        local train_2007_2012 = {}
        for i=1, #train_2007 do table.insert(train_2007_2012, train_2007[i]) end
        for i=1, #train_2012 do table.insert(train_2007_2012, train_2012[i]) end
        return  {
            train = train_2007_2012,
            test = loadRoiDataFn(projectDir .. 'data/proposals/selective_search_data/voc_2007_test.mat')
        }
    elseif str == 'test' then
        return  {
            test = loadRoiDataFn(projectDir .. 'data/proposals/selective_search_data/voc_2007_test.mat')
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
    elseif str == 'pascal_voc_2012' then
        return select_rois_pascal_2012(mode)
    elseif str == 'pascal_voc_2007_2012' then
        return select_rois_pascal_2007_2012(mode)
    elseif str == 'mscoco' then
        return select_rois_mscoco(mode)
    else
        error(('Invalid dataset: %s. Available datasets: pascal_voc_2007, pascal_voc_2012, pascal_voc_2007_2012 or mscoco'):format(name))
    end
end

------------------------------------------------------------------------------------------------------------

local function load_rois(name, mode)
--[[Load rois bboxes of all files into memory]]

    assert(name, 'Undefined dataset name: ' .. name)
    assert(mode == 'train' or mode == 'test', ('Invalid mode: %s. Valid modes: train or test.'):format(mode))

    local save_dir = paths.concat(projectDir, 'data/cache')

    if not paths.dirp(save_dir) then
        print('Creating cache dir: ' .. save_dir)
        os.execute('mkdir -p ' .. save_dir)
    end

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
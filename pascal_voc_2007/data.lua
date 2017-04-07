--[[
    Data loading method for the pascal voc 2007 dataset.
]]


--[[
local ffi = require 'ffi'
local dbc = require 'dbcollection.manager'
local string_ascii = require 'dbcollection.utils.string_ascii'
local ascii2str = string_ascii.convert_ascii_to_str
local pad = require 'dbcollection.utils.pad'
local unpad = pad.unpad_list
--]]

------------------------------------------------------------------------------------------------------------

local function fetch_data_set(set_name)
  
    local dbc = require 'dbcollection.manager'
    local string_ascii = require 'dbcollection.utils.string_ascii'
    local ascii2str = string_ascii.convert_ascii_to_str
    local pad = require 'dbcollection.utils.pad'
    local unpad = pad.unpad_list

    local dbloader = dbc.load{name='pascal_voc_2007', task='detection_light'}
  
    local loader = {}

    -- get image file path
    loader.getFilename = function(idx)
        local filename = ascii2str(dbloader:get(set_name, 'image_filenames', idx))[1]
        return paths.concat(dbloader.data_dir, filename)
    end

    -- get image ground truth boxes + class labels
    loader.getGTBoxes = function(idx)
        local objs_ids = unpad(dbloader:get(set_name, 'list_object_ids_per_image', idx):squeeze())
        if #objs_ids == 0 then
            return nil
        end
        local gt_boxes, gt_classes = {}, {}
        for _, id in ipairs(objs_ids) do
            local objID = dbloader:object(set_name, id + 1):squeeze()
            local bbox = dbloader:get(set_name, 'boxes', objID[3]):squeeze()
            local label = objID[2]
            table.insert(gt_boxes, bbox:totable())
            table.insert(gt_classes, label)
        end
        gt_boxes = torch.FloatTensor(gt_boxes)
        return gt_boxes,gt_classes
    end

    -- number of samples
    loader.nfiles = dbloader:size(set_name, 'image_filenames')[1]

    -- classes
    loader.classLabel = ascii2str(dbloader:get(set_name, 'classes'))

    return loader
end

------------------------------------------------------------------------------------------------------------

local function loader_train()
    return {
        train = fetch_data_set('trainval'),
        test = fetch_data_set('test')
    }
end

------------------------------------------------------------------------------------------------------------

local function loader_test()
    return {
        test = fetch_data_set('test')
    }
end

------------------------------------------------------------------------------------------------------------

local function data_loader(mode)
    assert(mode)

    if mode == 'train' then
        return function() return loader_train() end
    elseif mode == 'test' then
        return function() return loader_test() end
    else
        error(('Undefined mode: %s. mode must be either \'train\' or \'test\''):format(mode))
    end
end

------------------------------------------------------------------------------------------------------------

return data_loader
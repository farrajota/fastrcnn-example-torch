--[[
    Data loading method for the pascal voc 2007 dataset.
]]


local dbc = require 'dbcollection.manager'
local string_ascii = require 'dbcollection.utils.string_ascii'
local ascii2str = string_ascii.convert_ascii_to_str
local pad = require 'dbcollection.utils.pad'
local unpad = pad.unpad_list

------------------------------------------------------------------------------------------------------------

local function fetch_loader_mscoco(set_name)
    local dbloader = dbc.load{name='mscoco', task='detection_2015_d'}

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
            local bbox = dbloader:get(set_name, 'boxes', objID[6]):squeeze()
            local label = objID[4]
            table.insert(gt_boxes, bbox:totable())
            table.insert(gt_classes, label)
        end
        gt_boxes = torch.FloatTensor(gt_boxes)
        return gt_boxes,gt_classes
    end

    -- number of samples
    local nfiles = dbloader:size(set_name, 'image_filenames')[1]
    loader.nfiles = nfiles

    -- classes
    local class_names = ascii2str(dbloader:get(set_name, 'category'))
    loader.classLabel = class_names

    -- class ids (Only coco eval requires this)
    loader.classID = function(idx)
        return dbloader:get(set_name, 'category_id', idx):squeeze()
    end

    -- file ids (Only coco eval requires this)
    loader.fileID = function(idx)
        return dbloader:get(set_name, 'image_id', idx):squeeze()
    end

    return loader
end

------------------------------------------------------------------------------------------------------------

local function fetch_loader_pascal(set_name)

    local dbloader = dbc.load{name='pascal_voc_2007', task='detection_d'}

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
            local is_difficult = objID[5]
            if not (set_name == 'test' and is_difficult == 2) then
                table.insert(gt_boxes, bbox:totable())
                table.insert(gt_classes, label)
            end
        end
        gt_boxes = torch.FloatTensor(gt_boxes)
        return gt_boxes,gt_classes
    end

    -- number of samples
    local nfiles = dbloader:size(set_name, 'image_filenames')[1]
    loader.nfiles = nfiles

    -- classes
    local class_names = ascii2str(dbloader:get(set_name, 'classes'))
    loader.classLabel = class_names

    -- class ids (Only coco eval requires this)
    loader.classID = function(idx) return idx end

    -- file ids (Only coco eval requires this)
    loader.fileID = function(idx)
        return dbloader:get(set_name, 'image_id', idx):squeeze()
    end

    return loader
end

------------------------------------------------------------------------------------------------------------

local function fetch_loader_dataset(name, set_name)
    local str = string.lower(name)
    if str == 'pascal_voc_2007' then
        if set_name == 'train' then
            return fetch_loader_pascal('trainval')
        else
            return fetch_loader_pascal('test')
        end
    elseif str == 'mscoco' then
        if set_name == 'test' then
            return fetch_loader_mscoco('val')
        else
            return fetch_loader_mscoco('train')
        end
    else
        error(('Invalid dataset: %s. Available options: pascal_voc_2007 or mscoco.'):format(name))
    end
    return dbloader
end

------------------------------------------------------------------------------------------------------------

local function loader_train(name)
    return {
        train = fetch_loader_dataset(name, 'train'),
        test = fetch_loader_dataset(name, 'test')
    }
end

------------------------------------------------------------------------------------------------------------

local function loader_test(name)
    return {
        test = fetch_loader_dataset(name, 'test')
    }
end

------------------------------------------------------------------------------------------------------------

local function data_loader(name, mode)
    assert(mode)

    if mode == 'train' then
        return function() return loader_train(name) end
    elseif mode == 'test' then
        return function() return loader_test(name) end
    else
        error(('Undefined mode: %s. mode must be either \'train\' or \'test\''):format(mode))
    end
end

------------------------------------------------------------------------------------------------------------

return data_loader
--[[
    Data loading method for the pascal voc 2007 dataset.
]]


------------------------------------------------------------------------------------------------------------

local function get_db_loader(name)
    local dbc = require 'dbcollection.manager'
    local dbloader
    local str = string.lower(name)
    if str == 'pascal_voc_2007' then
        dbloader = dbc.load{name='pascal_voc_2007', task='detection_d'}
    elseif str == 'pascal_voc_2012' then
        dbloader = dbc.load{name='pascal_voc_2012', task='detection_d'}
    elseif str == 'mscoco' then
        dbloader = dbc.load{name='mscoco', task='detection_2015_d'}
    else
        error(('Undefined dataset: %s. Available options: pascal_voc_2007 or mscoco.'):format(name))
    end
    return dbloader
end

------------------------------------------------------------------------------------------------------------

local function fetch_loader_pascal_2007(set_name)

    local string_ascii = require 'dbcollection.utils.string_ascii'
    local ascii2str = string_ascii.convert_ascii_to_str
    local pad = require 'dbcollection.utils.pad'
    local unpad = pad.unpad_list

    -- get dataset loader
    local dbloader = get_db_loader('pascal_voc_2007')

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
            local objID = dbloader:object(set_name, id +1):squeeze() --id is 0-indexed, need +1
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

local function fetch_loader_pascal_2012(set_name)

    local string_ascii = require 'dbcollection.utils.string_ascii'
    local ascii2str = string_ascii.convert_ascii_to_str
    local pad = require 'dbcollection.utils.pad'
    local unpad = pad.unpad_list

    -- get dataset loader
    local dbloader = get_db_loader('pascal_voc_2012')

    local loader = {}

    if set_name == 'test' then
        set_name = 'val'
    end


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
            local objID = dbloader:object(set_name, id +1):squeeze() --id is 0-indexed, need +1
            local bbox = dbloader:get(set_name, 'boxes', objID[3]):squeeze()
            local label = objID[2]
            local is_difficult = objID[5]
            if not (set_name == 'val' and is_difficult == 2) then
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

local function fetch_loader_mscoco(set_name)

    local string_ascii = require 'dbcollection.utils.string_ascii'
    local ascii2str = string_ascii.convert_ascii_to_str
    local pad = require 'dbcollection.utils.pad'
    local unpad = pad.unpad_list

    -- get dataset loader
    local dbloader = get_db_loader('mscoco')

    local loader = {}

    if set_name == 'test' then
        set_name = 'val'
    end


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
            local objID = dbloader:object(set_name, id +1):squeeze() --id is 0-indexed, need +1
            local bbox = dbloader:get(set_name, 'boxes', objID[7]):squeeze()
            local label = objID[5]
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

local function fetch_loader_pascal_2007_2012(set_name)
--[[ Combine Pascal VOC 2007  + 2012 datasets ]]

    if set_name == 'train' then
        local loader_voc_2007 = fetch_loader_pascal_2007('trainval')
        local loader_voc_2012 = fetch_loader_pascal_2012('trainval')

        local loader = {}

        -- size datasets
        local size_voc_2007 = loader_voc_2007.nfiles
        local size_voc_2012 = loader_voc_2012.nfiles
        local nfiles_total = size_voc_2007 + size_voc_2012

        -- get image file path
        loader.getFilename = function(idx)
            if idx <= size_voc_2007 then
                return loader_voc_2007.getFilename(idx)
            else
                return loader_voc_2012.getFilename(idx - size_voc_2007)
            end
        end

        -- get image ground truth boxes + class labels
        loader.getGTBoxes = function(idx)
            if idx <= size_voc_2007 then
                return loader_voc_2007.getGTBoxes(idx)
            else
                return loader_voc_2012.getGTBoxes(idx - size_voc_2007)
            end
        end

        -- number of samples
        loader.nfiles = nfiles_total

        -- classes
        loader.classLabel = loader_voc_2007.classLabel

        return loader
    else
        return fetch_loader_pascal_2007(set_name)
    end

end

------------------------------------------------------------------------------------------------------------

local function fetch_loader_dataset(name, set_name)
    local str = string.lower(name)
    if str == 'pascal_voc_2007' then
        if set_name == 'train' then
            return fetch_loader_pascal_2007('trainval')
        else
            return fetch_loader_pascal_2007('test')
        end
    elseif str == 'pascal_voc_2012' then
        if set_name == 'train' then
            return fetch_loader_pascal_2012('train')
        else
            return fetch_loader_pascal_2012('val')
        end
    elseif str == 'pascal_voc_2007_2012' then
        if set_name == 'train' then
            return fetch_loader_pascal_2007_2012('train')
        else
            return fetch_loader_pascal_2007_2012('test')
        end
    elseif str == 'mscoco' then
        if set_name == 'test' then
            return fetch_loader_mscoco('val')
        else
            return fetch_loader_mscoco('train')
        end
    else
        error(('Invalid dataset: %s. Available datasets: pascal_voc_2007, pascal_voc_2012, pascal_voc_2007_2012 or mscoco'):format(name))
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
        error(('Invalid mode: %s. mode must be either \'train\' or \'test\''):format(mode))
    end
end

------------------------------------------------------------------------------------------------------------

return data_loader
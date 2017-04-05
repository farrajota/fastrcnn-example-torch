--[[
    Data loading method for the pascal voc 2007 dataset.
]]


local dbc = require 'dbcollection.manager'
local string_ascii = require 'dbcollection.utils.string_ascii'
local ascii2str = string_ascii.convert_ascii_to_str

------------------------------------------------------------------------------------------------------------

local function fetch_data_set(dbloader, set_name)
    local loader = {}

    -- get image file path
    loader.getFilename = function(idx)
        return return ascii2str(dbloader:get(set_name, 'image_filenames', idx))[1]
    end

    -- get image ground truth boxes + class labels
    loader.getGTBoxes = function(idx)
        local size = dataset.data.train.filenameList.objectIDList[idx]:size(1)
        local gt_boxes, gt_classes = {}, {}
        for i=1, size do
            local objID = dataset.data.train.filenameList.objectIDList[idx][i]
            if objID == 0 then
                break
            end
            local bbox = dataset.data.train.bbox[dataset.data.train.object[objID][3]]
            local label = dataset.data.train.object[objID][2]
            table.insert(gt_boxes, bbox:totable())
            table.insert(gt_classes, label)
        end
        gt_boxes = torch.FloatTensor(gt_boxes)
        return gt_boxes,gt_classes
    end,

    -- number of samples
    loader.nfiles =  dbloader:size(set_name, 'image_filenames')[1],

    -- classes
    loader.classLabel = ascii2str(dbloader:get(set_name, 'classes'))

    return loader
end

------------------------------------------------------------------------------------------------------------

local function loader_train()
    local dbloader = dbc.load('pascal_voc_2007')
    return {
        train = fetch_data_set(dbloader, 'trainval'),
        test = fetch_data_set(dbloader, 'test')
    }
end

------------------------------------------------------------------------------------------------------------

local function loader_test()
    local dbloader = dbc.load{name='pascal_voc_2007', task='no_difficult'}
    return {
        test = fetch_data_set(dbloader, 'test')
    }
end

------------------------------------------------------------------------------------------------------------

local function data_loader(mode)
    assert(mode)

    if mode == 'train' then
        return loader_train()
    elseif mode == 'test' then
        return loader_test()
    else
        error(('Undefined mode: %s. mode must be either \'train\' or \'test\''):format(mode))
    end
end

------------------------------------------------------------------------------------------------------------

return data_loader
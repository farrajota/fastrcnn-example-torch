--[[
    Download all data necessary for this repo in a single script.
]]


-- MS COCO annotations for all datasets
paths.dofile('download_eval_annotations.lua')

-- Imagenet pre-trained models
paths.dofile('download_pretrained_models.lua')

-- roi proposals
paths.dofile('download_roi_proposals.lua')
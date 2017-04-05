--[[
    Models list
]]


local function select_model(name, nGPU, nClasses, has_bbox_regressor)
    assert(name)
    assert(nGPU)
    assert(nClasses)
    assert(has_bbox_regressor)

    local model

    local str = string.lower(name)
    if string.match(str, 'alexnet') then
        model = require 'models.alexnet'
    elseif string.match(str, 'vgg') then
        model = require 'models.vgg'
    elseif string.match(str, 'zeiler') then
        model = require 'models.zeiler'
    elseif string.match(str, 'resnet') then
        model = require 'models.resnet'
    elseif string.match(str, 'inception') then
        model = require 'models.inceptionv3'
    else
        error('Undefined network type: ' .. name.. '. Available network types: alexnet, vgg, zeiler, resnet, inception.')
    end

    return model(nGPU, nClasses, has_bbox_regressor, str)
end

return select_model
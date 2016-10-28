--[[
    Models list
]]

local function parse(name)
    local str = string.lower(name)
    if string.match(str, 'alexnet') then
        return 'alexnet'
    elseif string.match(str, 'vgg') then
        return 'vgg'
    elseif string.match(str, 'zeiler') then
        return 'zeiler'
    elseif string.match(str, 'resnet') then
        return 'resnet'
    elseif string.match(str, 'inception')then
        return 'inceptionv3'
    else
        error('Undefined network type: ' .. name..'. Available network types: alexnet, vgg, zeiler, resnet, inception.')
    end
end


return {
    alexnet = paths.dofile('alexnet.lua'), --require 'models.alexnet',
    zeiler = paths.dofile('zeiler.lua'),
    vgg = paths.dofile('vgg.lua'),
    resnet = paths.dofile('resnet.lua'),
    inceptionv3 = paths.dofile('inceptionv3.lua'),
    parse = parse
}
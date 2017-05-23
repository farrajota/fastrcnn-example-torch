--[[
    Models list
]]


local function select_model(name, nGPU, nClasses)
    assert(name)
    assert(nGPU)
    assert(nClasses)

    local model

    local str = string.lower(name)
    if string.match(str, 'alexnet') then
        model = paths.dofile('alexnet.lua')
    elseif string.match(str, 'vgg') then
        model = paths.dofile('vgg.lua')
    elseif string.match(str, 'zeiler') then
        model = paths.dofile('zeiler.lua')
    elseif string.match(str, 'resnet') then
        model = paths.dofile('resnet.lua')
    elseif string.match(str, 'inception') then
        model = paths.dofile('inceptionv3.lua')
    else
        error('Invalid network type: ' .. name .. '. Available network types: alexnet, vgg, zeiler, resnet, inception.')
    end

    return model(nGPU, nClasses, str)
end

return select_model
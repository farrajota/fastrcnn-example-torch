--[[
    Download models and convert/store them to torch7 file format.
]]


require 'paths'
require 'torch'


-- download models
if not pcall(require,'download.download_alexnet') then
    print('Failed downloading the pretrained model: alexnet')
end
if not pcall(require, 'download.download_zeiler') then
    print('Failed downloading the pretrained model: zeiler')
end
if not pcall(require, 'download.download_vgg16_vgg19') then
    print('Failed downloading the pretrained model: vgg')
end
if not pcall(require, 'download.download_resnet') then
    print('Failed downloading the pretrained model: resnet')
end
if not pcall(require, 'download.download_googlenet') then
    print('Failed downloading the pretrained model: googlenet')
end

print('==> Downloads complete.')
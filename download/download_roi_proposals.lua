--[[
    Object proposals download for Fast-RCNN.
]]


require 'paths'
require 'torch'
paths.dofile('../projectdir.lua')


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 Fast-RCNN download roi proposals.')
cmd:text()
cmd:text(' ---------- General options ------------------------------------')
cmd:text()
cmd:option('-save_path',  projectDir .. '/data', 'Experiment ID')
cmd:text()

local opt = cmd:parse(arg or {})

local savepath = paths.concat(opt.save_path, 'proposals2')

-- create directory if needed
if not paths.dirp(savepath) then
    print('creating directory: ' .. savepath)
    os.execute('mkdir -p ' .. savepath)
end

-- download rois
print('==> Downloading region proposals to: ' .. savepath)

print('Downloading selective search proposals...')
if not paths.filep(paths.concat(savepath, 'selective_search_data.tgz')) then
    local url = 'http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/selective_search_data.tgz'
    local command = ('wget -O %s %s'):format(paths.concat(savepath, 'selective_search_data.tgz'), url)
    os.execute(command)
end

print('Downloading multiscale combinatorial grouping proposals...')

-- url 1
if not paths.filep(paths.concat(savepath, 'MCG-COCO-train2014-boxes.tgz')) then
    local url = 'https://data.vision.ee.ethz.ch/jpont/mcg/MCG-COCO-train2014-boxes.tgz'
    local command = ('wget -O %s %s'):format(paths.concat(savepath, 'MCG-COCO-train2014-boxes.tgz'), url)
    os.execute(command)
end

-- url 2
if not paths.filep(paths.concat(savepath, 'MCG-COCO-val2014-boxes.tgz')) then
    local url = 'https://data.vision.ee.ethz.ch/jpont/mcg/MCG-COCO-val2014-boxes.tgz'
    local command = ('wget -O %s %s'):format(paths.concat(savepath, 'MCG-COCO-val2014-boxes.tgz'), url)
    os.execute(command)
end

-- url 3
if not paths.filep(paths.concat(savepath, 'MCG-COCO-test2014-boxes.tgz')) then
    local url = 'https://data.vision.ee.ethz.ch/jpont/mcg/MCG-COCO-test2014-boxes.tgz'
    local command = ('wget -O %s %s'):format(paths.concat(savepath, 'MCG-COCO-test2014-boxes.tgz'), url)
    os.execute(command)
end

-- url 4
if not paths.filep(paths.concat(savepath, 'MCG-COCO-test2015-boxes.tgz')) then
    local url = 'https://data.vision.ee.ethz.ch/jpont/mcg/MCG-COCO-test2015-boxes.tgz'
    local command = ('wget -O %s %s'):format(paths.concat(savepath, 'MCG-COCO-test2015-boxes.tgz'), url)
    os.execute(command)
end

print('Extracting proposals...')
local command = ('tar -xvf %s'):format(paths.concat(savepath, 'selective_search_data.tgz'), savepath)
os.execute(command)
local extract_path = savepath
if not paths.dirp(extract_path) then
    print('creating directory: ' .. extract_path)
    os.execute('mkdir -p ' .. extract_path)
end
command = ('tar -xvf %s -C %s'):format(paths.concat(savepath, 'MCG-COCO-train2014-boxes.tgz'), extract_path)
os.execute(command)
command = ('tar -xvf %s -C %s'):format(paths.concat(savepath, 'MCG-COCO-val2014-boxes.tgz'), extract_path)
os.execute(command)
command = ('tar -xvf %s -C %s'):format(paths.concat(savepath, 'MCG-COCO-test2014-boxes.tgz'), extract_path)
os.execute(command)
command = ('tar -xvf %s -C %s'):format(paths.concat(savepath, 'MCG-COCO-test2015-boxes.tgz'), extract_path)
os.execute(command)

print('==> ROI proposals download/extracting complete.')
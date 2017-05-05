--[[
    Download Pascal VOC 2007 and MSCOCO annotations files for mAP evaluation when selecting the coco option.
]]


require 'paths'
require 'torch'
paths.dofile('../projectdir.lua')


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 Fast-RCNN download+extract annotation files.')
cmd:text()
cmd:text(' ---------- General options ------------------------------------')
cmd:text()
cmd:option('-save_path',  projectDir .. 'data/', 'Experiment ID')
cmd:text()

local opt = cmd:parse(arg or {})
local savepath = paths.concat(opt.save_path, 'coco_eval_annots')

-- create directory if needed
if not paths.dirp(savepath) then
    print('creating directory: ' .. savepath)
    os.execute('mkdir -p ' .. savepath)
end

print('==> Downloading Alexnet model... ')

local url1 = 'http://mscoco.org/static/annotations/PASCAL_VOC.zip'
local url2 = 'http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip'

-- file names
local filename_model1 = paths.concat(savepath, 'PASCAL_VOC.t7')
local filename_model2 = paths.concat(savepath, 'instances_train-val2014.t7')

-- download file
-- url1
print(filename_model1)
if not paths.filep(filename_model1) then
  local command = ('wget -O %s %s'):format(filename_model1, url1)
  os.execute(command)
end
-- url2
print(filename_model2)
if not paths.filep(filename_model2) then
  local command = ('wget -O %s %s'):format(filename_model2, url2)
  os.execute(command)
end


-- extract files
os.execute(('unzip %s -d %s'):format(filename_model1, savepath))
os.execute(('unzip %s -d %s'):format(filename_model2, savepath))

-- copy files from folders to the root savepath and delete them afterwards
if paths.dirp(paths.concat(savepath, 'PASCAL_VOC')) then
    os.execute(('cp %s/* %s'):format(paths.concat(savepath, 'PASCAL_VOC'), savepath))
    os.execute('rm -rf ' .. paths.concat(savepath, 'PASCAL_VOC'))
end
if paths.dirp(paths.concat(savepath, 'annotations')) then
    os.execute(('cp %s/* %s'):format(paths.concat(savepath, 'annotations'), savepath))
    os.execute('rm -rf ' .. paths.concat(savepath, 'annotations'))
end

print('Done.')
--[[
    Object proposals download for Fast-RCNN.
]]

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Region Of Interest (ROI) boxes pre-computed proposals download script.')
cmd:text()
cmd:text('Options:')
-- data sampling
cmd:option('-save_dir', '../data/proposals', 'Download models to this folder.')

-- parse options
local opt = cmd:parse(arg or {})

-- root folder to store pretrained models (or links)
local root_folder = '../data/proposals'

-- create directory if needed
print('creating directory: ' .. opt.save_dir)
os.execute('mkdir -p ' .. opt.save_dir)
os.execute('mkdir -p ' .. root_folder)

-- download models
local root_folder = '../data/proposals'
print('==> Downloading region proposals to: ' .. opt.save_dir)
if opt.download_model == 'all' or opt.download_model == 'ss' or opt.download_model == 'selectivesearch' then 
    print('Downloading selective search proposals...')
    if not paths.filep(paths.concat(opt.save_dir, 'selective_search_data.tgz')) then
        local url = 'http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/selective_search_data.tgz'
        local command = ('wget -O %s %s'):format(paths.concat(opt.save_dir, 'selective_search_data.tgz'), url)
        os.execute(command)
    end
    
    print('Downloading multiscale combinatorial grouping proposals...')
    
    -- url 1
    if not paths.filep(paths.concat(opt.save_dir, 'MCG-COCO-train2014-boxes.tgz')) then
        local url = 'https://data.vision.ee.ethz.ch/jpont/mcg/MCG-COCO-train2014-boxes.tgz'
        local command = ('wget -O %s %s'):format(paths.concat(opt.save_dir, 'MCG-COCO-train2014-boxes.tgz'), url)
        os.execute(command)
    end
        
    -- url 2
    if not paths.filep(paths.concat(opt.save_dir, 'MCG-COCO-val2014-boxes.tgz')) then
        local url = 'https://data.vision.ee.ethz.ch/jpont/mcg/MCG-COCO-val2014-boxes.tgz'
        local command = ('wget -O %s %s'):format(paths.concat(opt.save_dir, 'MCG-COCO-val2014-boxes.tgz'), url)
        os.execute(command)
    end
    
    -- url 3
    if not paths.filep(paths.concat(opt.save_dir, 'MCG-COCO-test2014-boxes.tgz')) then
        local url = 'https://data.vision.ee.ethz.ch/jpont/mcg/MCG-COCO-test2014-boxes.tgz'
        local command = ('wget -O %s %s'):format(paths.concat(opt.save_dir, 'MCG-COCO-test2014-boxes.tgz'), url)
        os.execute(command)
    end
    
    -- url 4
    if not paths.filep(paths.concat(opt.save_dir, 'MCG-COCO-test2015-boxes.tgz')) then
        local url = 'https://data.vision.ee.ethz.ch/jpont/mcg/MCG-COCO-test2015-boxes.tgz'
        local command = ('wget -O %s %s'):format(paths.concat(opt.save_dir, 'MCG-COCO-test2015-boxes.tgz'), url)
        os.execute(command)
    end
    
    print('Extracting proposals...')
    local command = ('tar -xvf %s -C %s'):format(paths.concat(opt.save_dir, 'selective_search_data.tgz'), opt.save_dir)
    os.execute(command)
    command = ('tar -xvf %s -C %s'):format(paths.concat(opt.save_dir, 'MCG-COCO-train2014-boxes.tgz'), paths.concat(opt.save_dir, 'MCG'))
    os.execute(command)
    command = ('tar -xvf %s -C %s'):format(paths.concat(opt.save_dir, 'MCG-COCO-val2014-boxes.tgz'), paths.concat(opt.save_dir, 'MCG'))
    os.execute(command)
    command = ('tar -xvf %s -C %s'):format(paths.concat(opt.save_dir, 'MCG-COCO-test2014-boxes.tgz'), paths.concat(opt.save_dir, 'MCG'))
    os.execute(command)
    command = ('tar -xvf %s -C %s'):format(paths.concat(opt.save_dir, 'MCG-COCO-test2015-boxes.tgz'), paths.concat(opt.save_dir, 'MCG'))
    os.execute(command)
    
    print('Done.')
end

print('==> ROI proposals download/extracting complete.')
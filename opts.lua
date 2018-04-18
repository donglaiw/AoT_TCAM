--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--
-- gpu ID
-- CUDA_VISIBLE_DEVICES
--
local M = { }
paths.dofile('util/misc.lua')
function M.parse(arg)
   local defaultDir = paths.concat('./')
   -- local defaultDir = paths.concat(os.getenv('HOME'), 'fbcunn_imagenet')

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Imagenet Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-cache',
               defaultDir ..'/imagenet_runs_oss',
               'subdirectory in which to save/log experiments')
    cmd:option('-data',
               defaultDir .. '/imagenet_raw_images/256',
               'Home of ImageNet dataset')
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-GPUs',             '1', 'Number of GPUs to use by default')
    cmd:option('-backend',     'cudnn', 'Options: cudnn | fbcunn | cunn')
    ------------- Data options ------------------------
    cmd:option('-dType',            'img@', 'img, mat')
    cmd:option('-nDonkeys',         12, 'number of donkeys to initialize (data loading threads)')
    cmd:option('-doMeanStd',        0, 'normalize input data')
    cmd:option('-dNum',             1, 'stride for input id')
    cmd:option('-labelOffset',      0, 'offset of label')
    cmd:option('-testSet',          'test' , 'test or val')
    cmd:option('-imFormat',         'image%s_%04d.jpg' , 'format of image names')
    cmd:option('-imType',          1 , '1: rgb, 2:_x_y flow')
    cmd:option('-cropType',          1 , '1: random, 2:five corner, 4:center')
    cmd:option('-xType',            5 , 'pre-procession type')
    cmd:option('-TcropType',        '1' , '1: random, 2:center, 3: start id')
    cmd:option('-loadSize',        '256,256' , 'load size')
    cmd:option('-sampleSize',      '224,224' , 'crop size')
    cmd:option('-nClasses',          101 , 'number of classes')
    ------------- Training options --------------------
    cmd:option('-iter_size',       1,    'Number of total epochs to run')
    cmd:option('-nEpochs',         55,    'Number of total epochs to run')
    cmd:option('-epochSize',       10000, 'Number of batches per epoch')
    cmd:option('-epochSave',       1, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       128,   'mini-batch size (1 = pure stochastic)')
    cmd:option('-batchRand',      0, '0: uniform across class, 1: uniform data')
    ------------- Test options --------------------
    cmd:option('-testRange',      '8,1', '1/8')
    cmd:option('-testNum',        -1, 'how many batches to test')
    cmd:option('-testOpt',         0, 'test opt')
    cmd:option('-testErr',         0, 'no error display')
    cmd:option('-testSave',        'f8' , 'which layer out')
    cmd:option('-testSaveId',       1, 'which module')
    cmd:option('-testOut',         '', 'where to save')
    cmd:option('-testReverse',      1, '1:normal, -1:reverse')
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    cmd:option('-paramId',  -1, 'learning rate schedule id')
    cmd:option('-optType',  'sgd', 'optimizer')
    cmd:option('-gradClip',         0, 'gradient clip range')
    cmd:option('-optOpt',         0, '0:sgd, 1:adam')
    ---------- Model options ----------------------------------
    cmd:option('-netType',     'cnn', 'Options: lstm | rcnn')
    cmd:option('-retrain',     'train', 'provide path to model to retrain with')
    cmd:option('-retrainOpt',   'train', 'retrain the last layer only')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:option('-Mdropout',  '', 'dropout for finetune')
    cmd:option('-conv1Type',  '0', 'replicate init')
    cmd:option('-wInit',         'kaiming', 'initialization method')
    cmd:option('-weightRatio',   1, 'gradient clip range')
    cmd:option('-lossType',      0, 'criterian type: sigmoid or softmax')
    -- video
    cmd:option('-vnF',     1, 'video: number of input frames')
    cmd:option('-vnC',     3, 'video: number of channels for each frame')
    -- temporal maxpool
    cmd:option('-timeMP',     '1@1', '')
    cmd:text()

    local opt = cmd:parse(arg or {})

    -- parameter
    opt.GPUs = utils.split(opt.GPUs,',')
    for i=1,#opt.GPUs do opt.GPUs[i] = tonumber(opt.GPUs[i]) end
    if opt.Mdropout ~= '' then
        local tmp = utils.split(opt.Mdropout,',')
        opt.Mdropout = {};
        for k,v in pairs(tmp) do
            opt.Mdropout[k] = tonumber(v);
        end
    else
        opt.Mdropout = {0.9,0.9}
    end
    opt.loadSize = utils.split(opt.loadSize,',')
    for i=1,#opt.loadSize do opt.loadSize[i] = tonumber(opt.loadSize[i]) end
    opt.sampleSize = utils.split(opt.sampleSize,',')
    for i=1,#opt.sampleSize do opt.sampleSize[i] = tonumber(opt.sampleSize[i]) end
    opt.nC = opt.vnC*opt.vnF -- total number of channel = number of frame * number of channel

    -- for testing
    opt.TcropType = utils.split(opt.TcropType,'@')
    for i=1,#opt.TcropType do opt.TcropType[i] = tonumber(opt.TcropType[i]) end
    opt.timeMP = utils.split(opt.timeMP,'@')
    for i=1,#opt.timeMP do opt.timeMP[i] = tonumber(opt.timeMP[i]) end
    opt.testRange = utils.split(opt.testRange,',')
    for i=1,#opt.testRange do opt.testRange[i] = tonumber(opt.testRange[i]) end

    -- load pretrain caffemodel
   if string.match(opt.retrain,"caffemodel") then
       opt.retrain = utils.split(opt.retrain,'@')
       assert(paths.filep(opt.retrain[1]), 'retrain caffe prototxt not found: ' .. opt.retrain[1])
       assert(paths.filep(opt.retrain[2]), 'retrain caffe model not found: ' .. opt.retrain[2])
   end

    -- model/result i/o 
    if opt.epochNumber==1 and opt.retrainOpt~='test' then --fresh start
        opt.save = opt.cache
        -- add date/time
        opt.save = paths.concat(opt.save, '_' .. os.date():gsub(' ',''):gsub(':','_'))
    else --reuse the same folder for old model
        if opt.retrainOpt=='test' then 
            -- get model id
            mid=string.sub(opt.retrain,opt.retrain:match'^.*()_'+1,-4)
            -- get model name
            if #opt.testOut==0 then
                opt.save = string.sub(opt.retrain,1,-4)..'_'
                opt.save = opt.save..opt.testSet..mid..'_'..opt.testRange[1]..'_'..opt.testRange[2]..'/'
            else
                opt.save = opt.testOut..opt.testSet..mid..'_'..opt.testRange[1]..'_'..opt.testRange[2]..'/'
            end
        else
            local ind=opt.retrain:match'^.*()/';
            opt.save = string.sub(opt.retrain,1,ind)
            opt.optimState= string.gsub(opt.retrain,'/model_','/optimState_')
        end
    end
    if not paths.dirp(opt.save) then
        os.execute('mkdir '..opt.save)
    end

    print(opt) -- output to the log
    return opt
end

return M

--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.GPUs[1]) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)

-- select model type
paths.dofile('model.lua')
paths.dofile('data_video.lua')
paths.dofile('trainCNN.lua')
paths.dofile('testCNN.lua')

epoch = opt.epochNumber
if opt.retrainOpt=='test' then
    test()
else
    for i=epoch,opt.nEpochs do
       train()
       epoch = epoch + 1
    end
end

--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'nn'
require 'cunn'
require 'cudnn'
local model_utils = require 'util.model_utils'
paths.dofile('trainUtil.lua')
require 'optim'

--[[
   1. Create Model
   2. Create Criterion
   3. If preloading option is set, preload weights from existing models appropriately
   4. Convert model to CUDA
]]--

-- 1. Create Network
--local config = opt.netType
local config = opt.netType .. '_' .. opt.backend
if opt.retrain == 'train' then
    paths.dofile('models/' .. config .. '.lua')
end
-- local dbg = require('debugger')

-- 2. Create Criterion
if opt.lossType==0 then -- binary classification
    criterion = nn.BCECriterion();
    opt.labelOffset=-1;
elseif opt.lossType==1 then -- multi-class
    criterion = nn.ClassNLLCriterion()
    opt.labelOffset=0;
end

-- 3. If preloading option is set, preload weights from existing models appropriately
if opt.retrain == 'train' then
    print('=> Creating model from file: models/' .. config .. '.lua')
    -- train from scratch
    if string.sub(opt.netType,1,2)=="sm" then
        cutorch.setDevice(opt.GPUs[1])
        model = createModel(opt.vnC,opt.nClasses) 
        if opt.netType=='smD' then
            model.modules[1].p = opt.Mdropout[1]
        end
    else
        model = createModel(opt.GPUs,opt.vnF*opt.vnC,opt.nClasses) -- for the model creation code, check the models/ folder
        -- change dropout rate
        if not string.match(opt.netType,'resnet') then --for vgg-fc-bn
            if string.match(opt.netType,'F_ucfP') then --for vgg-fc-bn
                model.modules[2].modules[5].p=opt.Mdropout[1];
                model.modules[2].modules[9].p=opt.Mdropout[2];
            end
            if string.match(opt.netType,'F_nin') then --for vgg-fc-bn
                model.modules[2].modules[4].p=opt.Mdropout[1];
                model.modules[2].modules[8].p=opt.Mdropout[2];
            end
        end
    end

   if opt.wInit == 'kaiming' then
     for indx,module in pairs(model:findModules('cudnn.SpatialConvolution')) do
       module.weight:normal(0,math.sqrt(2/(module.kW*module.kH*module.nOutputPlane)))
     end
   elseif opt.wInit == 'xavier' then
     for indx,module in pairs(model:findModules('cudnn.SpatialConvolution')) do
       module.weight:normal(0,math.sqrt(1/(module.kW*module.kH*module.nOutputPlane)))
     end
   end 
     for indx,module in pairs(model:findModules('Linear')) do
           modules.weight:copy(torch.randn(modules.weight:size()):mul(0.001)) 
           modules.bias:fill(0) -- small uniform numbers
     end
     if opt.nEpochs ==0 then -- save the initial model
           model:clearState()
           saveDataParallel(paths.concat(opt.save, 'model_0.t7'), model) -- defined in trainUtil.lua
           os.exit()
     end
else
    -- retrain 
   print('===> Loading model from file: ')
   print(opt.retrain)
   if opt.retrainOpt=='finetuneCBN' or opt.retrainOpt=='finetuneC' or opt.retrainOpt=='finetuneMD' or opt.retrainOpt=='finetuneMP' or opt.retrainOpt=='finetuneP' then
       require 'loadcaffe'
       model0 = loadcaffe.load(opt.retrain[1],opt.retrain[2],opt.retrain[3])
       -- local model0 = loadcaffe.load(opt.retrain[1],opt.retrain[2],opt.retrain[3])
       model_utils.caffeToTorch(model0,opt.retrainOpt,opt.Mdropout)
       if string.match(opt.netType,'PT') then -- regular
           model = createModel(opt.GPUs,opt.vnF,opt.vnC,nClasses,opt.timeMP) 
       else
           model = createModel(opt.GPUs,opt.vnF*opt.vnC,nClasses) 
       end
       -- copy from preload
       local numP1
       local offset=0;
       local countBN=0;
       if string.match(opt.netType, "pair2") then offset=-1;end
       if #opt.GPUs==1 then
           numP1 = #model.modules[1]['modules']
           for j=1,numP1 do
               if torch.type(model.modules[1]['modules'][j]):find('Convolution') then
                    model.modules[1]['modules'][j].weight:copy(model0.modules[j+offset-countBN].weight)
                    model.modules[1]['modules'][j].bias:copy(model0.modules[j+offset-countBN].bias)
                    if string.match(opt.netType,"bnC") then countBN = countBN+1;end
               end
           end
       else
           numP1 = #model.modules[1]['modules'][1]['modules']
           for j=1,numP1 do
               if torch.type(model.modules[1]['modules'][1]['modules'][j]):find('Convolution') then
                   --print(j,offset,numP1)
                   --print(#model.modules[1]['modules'][1]['modules'][j].weight)
                    model.modules[1]['modules'][1]['modules'][j].weight:copy(model0.modules[j+offset-countBN].weight)
                    model.modules[1]['modules'][1]['modules'][j].bias:copy(model0.modules[j+offset-countBN].bias)
                    if string.match(opt.netType,"bnC") then countBN = countBN+1;end
               end
           end
        end
       local numP2 = #model.modules[2]['modules']
       if (not string.match(opt.netType, "pair2")) and (not string.match(opt.netType, "nin"))  then
           for j=1,numP2 do
               if torch.type(model.modules[2]['modules'][j]):find('Linear') then
                    model.modules[2]['modules'][j].weight:copy(model0.modules[numP1+j].weight)
                    model.modules[2]['modules'][j].bias:copy(model0.modules[numP1+j].bias)
                elseif torch.type(model.modules[2]['modules'][j]):find('Dropout') then --dropout
                    model.modules[2]['modules'][j].p=model0.modules[numP1+j].p
               end
           end
       end
    elseif opt.retrainOpt=='test' or opt.retrainOpt=='finetune' or opt.retrainOpt=='finetuneL' or opt.retrainOpt=='finetuneM' or opt.retrainOpt == 'trainL'  or opt.retrainOpt == 'trainA' then
        -- finetune
        print('=> finetune or test from previous model: ' .. opt.netType)
        if type(opt.retrain)=="table" then
            require 'loadcaffe'
           -- finetune from caffe
           model = loadcaffe.load(opt.retrain[1],opt.retrain[2],opt.retrain[3])
           -- local dbg = require('debugger');dbg()
           -- local parameters, gradParameters = model:getParameters()
           -- modify last layer ...
           model_utils.caffeToTorch(model,opt.retrainOpt,opt.Mdropout)
       else
           -- finetune from t7
           assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
           model = loadDataParallel(opt.retrain,#opt.GPUs)
            -- change dropout rate
            if string.match(opt.netType,'F_ucfP') then --for vgg-fc-bn
                model.modules[2].modules[5].p=opt.Mdropout[1];
                model.modules[2].modules[9].p=opt.Mdropout[2];
             end
       end
   end
end

-- only last layer
if opt.retrainOpt == 'trainA' or opt.retrainOpt == 'trainL' or opt.retrainOpt == 'finetuneL' then
   print('=> only retrain last layer: ' .. opt.netType)
   local last_id = 22; -- alexnet
   if string.match(opt.netType, "vggbnCF_nin2pair2") then
       --AoT need to change last layer
       if opt.retrainOpt ~= 'trainA' then 
           -- first conv
            for j=1,#model.modules[1].modules[1].modules do
               if torch.type(model.modules[1].modules[1].modules[j]):find('Linear') or torch.type(model.modules[1].modules[1].modules[j]):find('Convolution') then
                   print('Freeze conv-layers: '..j..','..torch.type(model.modules[1].modules[1].modules[j]))
                    model.modules[1].modules[1].modules[j].accGradParameters = function() end
                    model.modules[1].modules[1].modules[j].updateParameters = function() end
                end
            end
           -- fc
            if opt.netType ~= "vggbnCF_nin2pair2fc" then
                for j=1,#model.modules[2].modules do
                   if torch.type(model.modules[2].modules[j]):find('Linear') or torch.type(model.modules[2].modules[j]):find('Convolution') then
                        print('Freeze fc-layers: '..j..','..torch.type(model.modules[2].modules[j]))
                        model.modules[2].modules[j].accGradParameters = function() end
                        model.modules[2].modules[j].updateParameters = function() end
                    end
                end
            end
        end
        -- first time, change the sturcture
        -- should move to model_utils
        if string.match(opt.retrainOpt , 'train') then
            if opt.netType == "vggbnCF_nin2pair2" or opt.netType == "vggbnCF_nin2pair2fc" then
                -- dropout
                model.modules[2].modules[5].p = opt.Mdropout[1]
                model.modules[2].modules[9].p = opt.Mdropout[2]
                print('change dropout: ',opt.Mdropout)
                model.modules[2].modules[12]=nn.Linear(1024, opt.nClasses)
                model.modules[2].modules[12].weight:copy(torch.randn(model.modules[2].modules[12].weight:size()):mul(0.001)) 
                model.modules[2].modules[12].bias:fill(0) -- small uniform numbers
                model.modules[2].modules[13]=nn.LogSoftMax()
            elseif opt.netType == "vggbnCF_nin2pair2NP" then
                model.modules[2].modules[10]=nn.View(-1,200704)
                model.modules[2].modules[11]=nn.Linear(200704, opt.nClasses)

                model.modules[2].modules[11].weight:copy(torch.randn(model.modules[2].modules[11].weight:size()):mul(0.001)) 
                model.modules[2].modules[11].bias:fill(0) -- small uniform numbers
                model.modules[2].modules[12]=nn.LogSoftMax()
                model.modules[2].modules[13]=nil
            elseif opt.netType == "resnetbnCF_nin2pair2" then
                model.modules[2].modules[5]=nn.Linear(2048, opt.nClasses)
                model.modules[2].modules[5].weight:copy(torch.randn(model.modules[2].modules[5].weight:size()):mul(0.001)) 
                model.modules[2].modules[5].bias:fill(0) -- small uniform numbers
                model.modules[2].modules[6]=nn.LogSoftMax()
            end
        end
   else -- no need to change structure
       if #opt.GPUs==1 then -- for imNet
           if opt.retrainOpt ~= 'trainA' then 
               if string.match(opt.netType, "vgg_ucfP") then
                   last_id=38
                   if opt.netType=="vgg_ucfPp5" or opt.netType=="vgg_ucfPfc" then
                       last_id=30
                   end
               end
               print('Freeze layers: '..opt.netType..','..last_id)
               for j=1,last_id do
                   if torch.type(model.modules[j]):find('Linear') or torch.type(model.modules[j]):find('Convolution') then
                        model.modules[j].accGradParameters = function() end
                        model.modules[j].updateParameters = function() end
                    end
                end
            end
        else -- for rand
           if opt.retrainOpt ~= 'trainA' then 
                for j=1,#model.modules[1].modules[1].modules do
                   if torch.type(model.modules[1].modules[1].modules[j]):find('Linear') or torch.type(model.modules[1].modules[1].modules[j]):find('Convolution') then
                        model.modules[1].modules[1].modules[j].accGradParameters = function() end
                        model.modules[1].modules[1].modules[j].updateParameters = function() end
                    end
                end
               -- fc
                if not string.match(opt.netType, "ucfPfc") then
                    for j=1,#model.modules[2].modules-2 do
                       if torch.type(model.modules[2].modules[j]):find('Linear') or torch.type(model.modules[2].modules[j]):find('Convolution') then
                            model.modules[2].modules[j].accGradParameters = function() end
                            model.modules[2].modules[j].updateParameters = function() end
                        end
                    end
                end
           end
       end
    end
end


if opt.backend == 'cudnn' then
  model = model:cuda()
  cudnn.convert(model, cudnn)
elseif opt.backend == 'cunn' then
  model = model:cuda()
elseif opt.backend ~= 'nn' then
  error'Unsupported backend'
end
if opt.backend ~= 'nn' then
    criterion:cuda()
end
collectgarbage()

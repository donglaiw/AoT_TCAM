--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'
paths.dofile('util/ffi_helper.lua')
paths.dofile('trainUtil.lua')

local batchNumber
local top1_epoch, loss_epoch
-- 2. SGD parameters
-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}
if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

-- print(model.modules)
local optimator = nn.Optim(model, optimState)

-- 3. train CNN with sgd
-- this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk

function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   local params, newRegime = paramsForEpoch(epoch,opt.paramId)
   if newRegime then
       -- Zero the momentum vector by throwing away previous state.
       --optimator = nn.Optim(model, optimState)
   optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }
   end
   optimator:setParameters(params)

   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()
   -- model:cuda() -- get it back on the right GPUs.
   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0
   for i=1,opt.epochSize*opt.iter_size do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, labels = trainLoader:sample(opt.batchSize)
            -- return sendTensor(inputs), sendTensor(labels)
            return inputs[1],inputs[2], labels
         end,
         -- the end callback (runs in the main thread)
         -- trainBatchRCNN
         trainBatchRCNN_p
      )
   end
   donkeys:synchronize()
   cutorch.synchronize()
   top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
   loss_epoch = loss_epoch / opt.epochSize
   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1_epoch,
      ['avg loss (train set)'] = loss_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t',
                       epoch, tm:time().real, loss_epoch, top1_epoch))
   print('\n')
   -- save model
   collectgarbage()
   -- sanitize(model)
   -- torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
   model:clearState()
   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
   --[[
   --]]
end -- of train()
-------------------------------------------------------------------------------------------
-- create tensor buffers in main thread and deallocate their storages.
-- the thread loaders will push their storages to these buffers when done loading
-- local inputsCPU = torch.FloatTensor()
-- local labelsCPU = torch.LongTensor()
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local inputs_roi = torch.CudaTensor()
local inputs_last = torch.CudaTensor()
local labels = torch.CudaTensor()
local timer = torch.Timer()
local dataTimer = torch.Timer()
local parameters, gradParameters = model:getParameters()

local function feval_p()
  return criterion.output, gradParameters
end
local top1_p=0;
local err1_p=0;
local dataLoadingTime_p=0;

function trainBatchRCNN_p(inputsCPU, inputs_roiCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage();collectgarbage()
   if batchNumber % opt.iter_size == 0 then
       model:zeroGradParameters()
       dataLoadingTime_p  = dataTimer:time().real
       timer:reset()
       top1_p=0;err1_p=0;
   end
   -- number of input is the same
   if batchNumber == 0 then 
       inputs:resize(inputsCPU:size()):copy(inputsCPU)
   else
       inputs:copy(inputsCPU)
   end
   -- if test, N-boxes per image
   -- need to resize labels
    inputs_roi:resize(inputs_roiCPU:size()):copy(inputs_roiCPU)
   if inputs_roiCPU:size(1)~=labelsCPU:size(1) then
       labels:resize(inputs_roi:size(1))
       for i=1,inputs_roi:size(1) do
           labels[i]=labelsCPU[inputs_roi[i][1]]
       end
   else
       labels:resize(labelsCPU:size()):copy(labelsCPU)
   end
    --[[
   im = inputsCPU:narrow(2,1,1):clone():fill(0)
   for i=1,inputs_roiCPU:size(1) do
       im[{{i,i},{},{inputs_roiCPU[i][3],inputs_roiCPU[i][5]},{inputs_roiCPU[i][2],inputs_roiCPU[i][4]}}]:fill(1);
   end
   local matio = require 'matio'; matio.save('db.mat',{a=inputsCPU,b=inputs_roiCPU,c=im,d=labelsCPU})
   local dbg = require('debugger');dbg()
   --]]
   --
   local outputs = model:forward({inputs,inputs_roi})
   local err = criterion:forward(outputs, labels)
   local gradOutputs = criterion:backward(outputs, labels)
   -- print(gradOutputs)
   model:backward({inputs,inputs_roi}, gradOutputs)
   -- print(torch.mean(gradParameters)..','..torch.min(gradParameters)..','..torch.max(gradParameters))
   -- print(#gradParameters)
   cutorch.synchronize()
   collectgarbage();collectgarbage();

   -- top-1 error
   local _,prediction_sorted = outputs:float():sort(2, true) -- descending
      for i=1,opt.batchSize do
         if prediction_sorted[i][1] == labelsCPU[i] then 
            top1_p = top1_p + 1;
         end
      end
    err1_p = err1_p + err

   if (batchNumber+1) % opt.iter_size == 0 then -- update
       gradParameters:div(opt.iter_size)
       optim.sgd(feval_p, parameters, optimState)
   end
   if (batchNumber+1) % opt.iter_size == 0 then -- display
       -- Calculate top-1 error, and print information
        top1_epoch = top1_epoch + top1_p/opt.iter_size; 
        top1_p = top1_p * 100 / opt.batchSize/opt.iter_size;
        err1_p = err1_p/opt.iter_size;
        print(('Epoch: [%d][%d-%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f LR %.0e DTime %.3f'):format(
              epoch, math.ceil((1+batchNumber)/opt.iter_size), batchNumber%opt.iter_size, opt.epochSize, timer:time().real, err1_p, top1_p,
              optimState.learningRate, dataLoadingTime_p))
       dataTimer:reset()
       loss_epoch = loss_epoch + err1_p
   end
   batchNumber = batchNumber + 1
end

function trainBatchRCNN(inputsCPU, inputs_roiCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()
   -- set the data and labels to the main thread tensor buffers (free any existing storage)
   -- receiveTensor(inputsThread, inputsCPU)
   -- receiveTensor(labelsThread, labelsCPU)

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   inputs_roi:resize(inputs_roiCPU:size()):copy(inputs_roiCPU)
   -- N-boxes per image
   -- need to resize labels
   if inputs_roiCPU:size(1)~=labelsCPU:size(1) then
       labels:resize(inputs_roi:size(1))
       for i=1,inputs_roi:size(1) do
           labels[i]=labelsCPU[inputs_roi[i][1]]
       end
   else
       labels:resize(labelsCPU:size()):copy(labelsCPU)
   end
   --[[
   im = inputsCPU:narrow(2,1,1):clone():fill(0)
   for i=1,inputs_roiCPU:size(1) do
       im[{{i,i},{},{inputs_roiCPU[i][3],inputs_roiCPU[i][5]},{inputs_roiCPU[i][2],inputs_roiCPU[i][4]}}]:fill(1);
   end
   local matio = require 'matio'; matio.save('db.mat',{a=inputsCPU,b=inputs_roiCPU,c=im,d=labelsCPU})
   local dbg = require('debugger');dbg()
   --]]
   local err, outputs
   feval = function(x)
      model:zeroGradParameters() -- have to have gpu1
      outputs = model:forward({inputs,inputs_roi})
      err = criterion:forward(outputs, labels)
      local gradOutputs = criterion:backward(outputs, labels)
      model:backward({inputs,inputs_roi}, gradOutputs)
      return err, gradParameters
   end
   optim.sgd(feval, parameters, optimState)

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   -- top-1 error
   local top1 = 0
   do
      local _,prediction_sorted = outputs:float():sort(2, true) -- descending
      for i=1,labels:size(1) do
         if prediction_sorted[i][1] == labels[i] then 
            top1_epoch = top1_epoch + 1; 
            top1 = top1 + 1
         end
      end
      top1 = top1 * 100 / labels:size(1);
   end

   -- Calculate top-1 error, and print information
   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f LR %.0e DTime %.3f'):format(
          epoch, batchNumber, opt.epochSize, timer:time().real, err, top1,
          optimState.learningRate, dataLoadingTime))
   --[[
   print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f LR %.0e DTime %.3f M %.5f, B %.4f'):format(
          epoch, batchNumber, opt.epochSize, timer:time().real, err, top1,
          optimState.learningRate, dataLoadingTime,torch.mean(model.modules[39].weight),torch.std(model.modules[39].weight)))
  --]]

   dataTimer:reset()
end

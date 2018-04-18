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


-- 3. train CNN with sgd
-- this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk

function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)
   local params, newRegime = paramsForEpoch(epoch,opt.paramId)
   -- constant update, good for finetune, otherwise hard to change lr
   optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }

      batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()
   -- model:evaluate()
   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0
   if opt.retrainOpt~='1' then
       for i=1,opt.epochSize*opt.iter_size do
          -- queue jobs to data-workers
          donkeys:addjob(
             -- the job callback (runs in data-worker thread)
             function()
                local inputs, labels = trainLoader:sample(opt.batchSize)
                -- return sendTensor(inputs), sendTensor(labels)
                return inputs, labels
             end,
             -- the end callback (runs in the main thread)
             trainBatchCNN_p
             -- trainBatchCNN
          )
       end
   else
        for i=1,opt.epochSize do
          -- queue jobs to data-workers
          donkeys:addjob(
             -- the job callback (runs in data-worker thread)
             function()
                local inputs, labels = trainLoader:sample(opt.batchSize)
                return sendTensor(inputs), sendTensor(labels)
             end,
             -- the end callback (runs in the main thread)
             trainBatchCNN_last
          )
      end
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
   if epoch % opt.epochSave == 0 then
       model:clearState()
       saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in trainUtil.lua
       torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
   end
end -- of train()
-------------------------------------------------------------------------------------------
-- create tensor buffers in main thread and deallocate their storages.
-- the thread loaders will push their storages to these buffers when done loading
-- local inputsCPU = torch.FloatTensor()
-- local labelsCPU = torch.LongTensor()
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()
local timer = torch.Timer()
local dataTimer = torch.Timer()
local parameters, gradParameters = model:getParameters()

local function feval_p()
  return criterion.output, gradParameters
end
top1_p=0;
err1_p=0;
dataLoadingTime_p=0;


function trainBatchCNN_p(inputsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()
   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)
   -- key: batchNumber is in serial
   if batchNumber % opt.iter_size == 0 then
       model:zeroGradParameters()
       dataLoadingTime_p  = dataTimer:time().real
       timer:reset()
       top1_p=0;err1_p=0;
   end
   local err, outputs
   outputs = model:forward(inputs)
   err = criterion:forward(outputs, labels)

   local gradOutputs = criterion:backward(outputs, labels)
   model:backward(inputs, torch.mul(gradOutputs,opt.weightRatio))
   cutorch.synchronize()

   local pred_sorted
   if outputs:size(2)==1 then
       pred_sorted = outputs:float():gt(0.5)
   else
       _,pred_sorted = outputs:float():sort(2, true) -- descending
   end
      for i=1,opt.batchSize do
         if pred_sorted[i][1] == labelsCPU[i] then 
            top1_p = top1_p + 1;
         end
      end
    err1_p = err1_p + err

   if (batchNumber+1) % opt.iter_size == 0 then -- update
       gradParameters:div(opt.iter_size)
       print('after:',gradParameters:min(),gradParameters:max())
       if opt.gradClip>0 then
           gradParameters:clamp(-opt.gradClip,opt.gradClip)
       end
       if opt.optOpt==0 then
           optim.sgd(feval_p, parameters, optimState)
       elseif opt.optOpt==1 then
           optim.adam(feval_p, parameters, optimState)
       end
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


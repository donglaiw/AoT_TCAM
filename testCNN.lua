--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local testDataIterator = function()
   testLoader:reset()
   return function() return testLoader:get_batch(false) end
end
matio = require 'matio'
batchNum = math.ceil(nTest/opt.batchSize)
if opt.testNum~=-1 then -- only a few batches
    batchNum=opt.testNum
end
local top1_center, loss
local timer = torch.Timer()

function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)
   cutorch.synchronize()
   timer:reset()
   -- set the dropouts to evaluate mode
   -- model:training()
   model:evaluate()
   top1_center = 0
   top1_count = 0
   loss = 0
   local saveSuf='.mat'
   if opt.testSave=='f7' then
       saveSuf='_f7.mat'
   end
   local i;
   -- opt.testReverse=-1
   for ii=1,batchNum,math.abs(opt.testReverse) do -- nTest is set in 1_data.lua
       if opt.testReverse<0 then
           i=batchNum+1-ii
       else
           i=ii;
       end
        if not paths.filep(opt.save..'test_'..i..saveSuf) then
          --  print('do:'..i)
          local indexStart = (i-1) * opt.batchSize + 1
          local indexEnd = math.min(indexStart + opt.batchSize - 1, nTest)
          if string.match(opt.netType, "RCNN") then
              donkeys:addjob(
                 function()
                    local inputs, labels = testLoader:get(indexStart, indexEnd)
                    return inputs[1],inputs[2], labels, i
                 end,
                 testBatchRCNN
              )
          else
              donkeys:addjob(
                 function()
                    local inputs, labels = testLoader:get(indexStart, indexEnd)
                    return inputs, labels, i
                 end,
                 testBatchCNN
              )
          end
      end
  end
   donkeys:synchronize()
   cutorch.synchronize()

   top1_center = top1_center * 100 / top1_count
   loss = loss / nTest
   testLogger:add{
      ['% top1 accuracy (test set) (center crop)'] = top1_center,
      ['avg loss (test set)'] = loss
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy [Center](%%):\t top-1 %.2f\t ',
                       epoch, timer:time().real, loss, top1_center))

   print('\n')


end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local inputs_roi = torch.CudaTensor()
local labels = torch.CudaTensor()

function testBatchCNN(inputsCPU, labelsCPU, batchNumber)
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)
   local outputs = model:forward(inputs)
   local pred = outputs:float()
   cutorch.synchronize()
   if opt.testDisplay==1 then
       local err = criterion:forward(outputs, labels)
       loss = loss + err*outputs:size(1)
       local top1 = 0
       local pred_sorted
       if outputs:size(2)==1 then
           pred_sorted = outputs:float():gt(0.5)
       else
           _,pred_sorted = outputs:float():sort(2, true) -- descending
       end
       for i=1,pred:size(1) do
          local g = labelsCPU[i]
          if pred_sorted[i][1] == g then 
              top1_center = top1_center + 1 
              top1 = top1 + 1
          end
       end
       top1 = top1 * 100 / pred:size(1);
       top1_count = top1_count + pred:size(1) 
       print(('Epoch: [%d][%d/%d]\t Err %.4f Top1-%%: %.2f'):format(epoch, batchNumber, batchNum, err, top1))
   else
       print(('Epoch: [%d][%d/%d]\t' ):format(epoch, batchNumber, batchNum))
   end
   if opt.testSave=='f8' then
       matio.save(opt.save..'test_'..batchNumber..'.mat',{pred=pred})
   elseif opt.testSave=='f7' then
       matio.save(opt.save..'test_'..batchNumber..'_f7.mat',{pred=pred,f7=model.modules[2].modules[opt.testSaveId].output:float()})
   end
end

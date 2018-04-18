
-- 1. Create loggers.

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
trainLogger=nil 
function paramsForEpoch(epoch,paramId)
    trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {}
    if paramId==0 then
        regimes={
            -- start, end,    LR,   WD,
            {  1,     18,   1e-2,   5e-4,1 },
            { 19,     29,   5e-3,   5e-4,1  },
            { 30,     43,   1e-3,   0,1 },
            { 44,     52,   5e-4,   0,1 },
            { 53,    1e8,   1e-4,   0,1 },
        }
    elseif paramId==1 then
        regimes={
            -- start, end,    LR,   WD,
            {  1,     3,   1e-3,   0,1 },
            {  4,     6,   5e-4,   0,1 },
            { 7,     9,   1e-4,   0,1 },
        }
    elseif paramId==2 then
        regimes={
            -- start, end,    LR,   WD,
            {  1,     5,   1e-3,   0,1},
            { 6,     10,   5e-4,   0,1 },
            { 11,     15,   1e-4,   0,1 },
            { 16,     20,   1e-4,   0,1 },
            { 21,     25,   1e-4,   0,1 },
        }
    elseif paramId==3 then
        -- for char-nn
        regimes={
            -- start, end,    LR,   WD,
            {  1,     10,   2e-3,   0 , 1},
            {  11,     50,   2e-3,   0.03,1},
        }
    elseif paramId==4 then
        -- for finetune vgg-rgb
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     1,   1e-3,   0 , 0},
            {  2,     2,   1e-4,   0 , 0},
            {  3,     3,   1e-5,   0 , 0},
        }
    elseif paramId==5 then
        -- for finetune vgg-flo
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     5,   1e-3,   5e-4 , 0},
            {  6,     100,   1e-4,   5e-4 , 0},
        }
    elseif paramId==5.1 then
        -- scrach vgg-flo
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     1,   1e-2,   5e-4 , 0},
            {  2,     100,   1e-3,   5e-4 , 0},
        }
    elseif paramId==5.2 then
        -- scrach vgg-flo
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     2,   1e-2,   5e-4 , 0},
            {  3,     100,   1e-3,   5e-4 , 0},
        }
    elseif paramId==5.22 then
        -- scrach vgg-flo
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     10,   1e-2,   5e-4 , 0},
            {  11,     100,   1e-3,   5e-4 , 0},
        }
    elseif paramId==5.222 then
        -- scrach vgg-flo
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     10000,   1e-2,   5e-4 , 0},
        }
    -- 5.*1: no wd
    elseif paramId==5.21 then
        -- for nEpoch=5
        -- no weight decay [regularization] for finetuning
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     2,   1e-2,   0 , 0},
            {  3,     100,   1e-3,   0 , 0},
        }
    elseif paramId==5.41 or paramId==5.211 then
        -- for nEpoch=10
        -- no weight decay [regularization] for finetuning
        --
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     5,   1e-2,   0 , 0},
            {  6,     100,   1e-3,   0 , 0},
        }
    elseif paramId==5.212 then
        -- for nEpoch=10
        -- no weight decay [regularization] for finetuning
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     10,   1e-2,   0 , 0},
            {  11,     100,   1e-3,   0 , 0},
        }
    elseif paramId==5.213 then
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     20,   1e-2,   0 , 0},
            {  21,     100,   1e-3,   0 , 0},
        }
    elseif paramId==5.214 then
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     10000,   1e-2,   0 , 0},
        }
    elseif paramId==5.31 then
        -- no weight decay [regularization] for finetuning
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     2,   1e-3,   0 , 0},
            {  3,     100,   1e-4,   0 , 0},
        }
    elseif paramId==5.311 then
        -- no weight decay [regularization] for finetuning
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     20,   1e-4,   0 , 0},
            {  21,     100,   1e-5,   0 , 0},
        }

    elseif paramId==5.312 then
        -- no weight decay [regularization] for finetuning
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     20,   1e-3,   0 , 0},
            {  21,     100,   1e-4,   0 , 0},
        }
    elseif paramId==5.3 then
        -- scrach vgg-flo
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     2,   1e-1,   5e-3 , 0},
            {  3,     5,   1e-2,   5e-4 , 0},
        }
    elseif paramId==6 then
        -- for finetune vgg-flo
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     5,   5e-3,   5e-4 , 0},
            {  6,     200,   5e-4,   5e-4 , 0},
        }
    elseif paramId==6.01 then
        -- for finetune vgg-flo
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     10,   5e-3,   5e-4 , 0},
            {  11,     200,   5e-4,   5e-4 , 0},
        }
    elseif paramId==6.1 then
        -- for finetune vgg-flo
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     100000,   5e-4,   5e-5 , 0},
        }
    elseif paramId==6.2 then
        -- for finetune vgg-flo
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     10000,   1e-4,   1e-5 , 0},
        }
    elseif paramId==6.3 then
        -- for finetune vgg-flo
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     10000,   1e-5,   1e-6 , 0},
        }
    elseif paramId==6.4 then
        -- for finetune vgg-flo
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     10000,   1e-6,   1e-7 , 0},
        }
    elseif paramId==7 then
        -- lstm seq10
        regimes={
            -- start, end,    LR,   WD, LRD
            {  1,     10000,   1e-1,   0 , 0},
        }

    end
    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then

            return { learningRate=row[3], weightDecay=row[4], learningRateDecay=row[5] }, epoch == row[1]
        end
    end
end

-- clear the intermediate states in the model before saving to disk
-- this saves lots of disk space
function sanitize(net)
  local list = net:listModules()
  for _,val in ipairs(list) do
        for name,field in pairs(val) do
           if torch.type(field) == 'cdata' then val[name] = nil end
           if name == 'homeGradBuffers' then val[name] = nil end
           if name == 'input_gpu' then val['input_gpu'] = {} end
           if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
           if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
           if (name == 'output' or name == 'gradInput') then
              -- val[name] = field.new()
              val[name] = nil
           end
        end
  end
end

function makeDataParallel(model, nGPU)
   if nGPU > 1 then
      print('converting module to nn.DataParallelTable')
      local model_single = model
      model = nn.DataParallelTable(1)
      for i=1, nGPU do
         cutorch.setDevice(opt.GPUs[i])
         model:add(model_single:clone():cuda(), i)
      end
   end
   cutorch.setDevice(opt.GPUs[1])
   return model
end
local function cleanDPT(module)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   local newDPT = nn.DataParallelTable(1)
   cutorch.setDevice(opt.GPUs[1])
   newDPT:add(module:get(1), opt.GPUs[1])
   return newDPT
end

function saveDataParallel(filename, model)
   if torch.type(model) == 'nn.DataParallelTable' then
      torch.save(filename, cleanDPT(model))
   elseif torch.type(model) == 'nn.Sequential' then
      local temp_model = nn.Sequential()
      for i, module in ipairs(model.modules) do
         if torch.type(module) == 'nn.ParallelTable' then
            local temp_model2 = nn.ParallelTable()
            for i2, module2 in ipairs(module.modules) do
                 if torch.type(module2) == 'nn.DataParallelTable' then
                    temp_model2:add(cleanDPT(module2))
                 else
                    temp_model2:add(module2)
                 end
            end
            temp_model:add(temp_model2)
         elseif torch.type(module) == 'nn.DataParallelTable' then
            temp_model:add(cleanDPT(module))
         else
            temp_model:add(module)
         end
      end
      torch.save(filename, temp_model)
   else
      error('This saving function only works with Sequential or DataParallelTable modules.')
   end
end

function loadDataParallel(filename, nGPU)
   if opt.backend == 'cudnn' then
      require 'cudnn'
   end
   -- only change conv part
   local model = torch.load(filename)
   if torch.type(model) == 'nn.DataParallelTable' then
      return makeDataParallel(model:get(1):float(), nGPU)
   elseif torch.type(model) == 'nn.Sequential' then
      if torch.type(model.modules[1]) == 'nn.ParallelTable' then
         for i,module in ipairs(model.modules[1].modules) do
             if torch.type(module) == 'nn.DataParallelTable' then
                model.modules[1].modules[i] = makeDataParallel(module:get(1):float(), nGPU)
             end
          end
      else
          for i,module in ipairs(model.modules) do
             if torch.type(module) == 'nn.DataParallelTable' then
                model.modules[i] = makeDataParallel(module:get(1):float(), nGPU)
             end
          end
      end
	  model:cuda()
      return model
   else
      error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end

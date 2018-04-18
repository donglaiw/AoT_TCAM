-- adapted from https://github.com/wojciechz/learning_to_execute
-- utilities for combining/flattening parameters in a model
-- the code in this script is more general than it needs to be, which is 
-- why it is kind of a large

require 'torch'
local model_utils = {}
require 'cunn'

function makeDataParallel(model, nGPU)
   if nGPU > 1 then
      print('converting module to nn.DataParallelTable')
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      if opt.backend == 'cudnn' and opt.cudnnAutotune == 1 then
        local gpu_table = torch.range(1, nGPU):totable()
        local dpt = nn.DataParallelTable(1, true):add(model, gpu_table):threads(function() require 'cudnn'
           cudnn.benchmark = true  end)
        dpt.gradInput = nil
        model = dpt:cuda()
      else
        local model_single = model
        model = nn.DataParallelTable(1)
        for i=1, nGPU do
           cutorch.setDevice(i)
           model:add(model_single:clone():cuda(), i)
        end
        cutorch.setDevice(opt.GPU)
      end
   else
      if (opt.backend == 'cudnn' and opt.cudnnAutotune == 1) then
        require 'cudnn'
        cudnn.benchmark = true
      end
   end
   return model
end

function model_utils.caffeToTorch(tmp_model,tmp_opt,tmp_dr)
    if tmp_opt=='finetuneMD' or tmp_opt=='finetuneRD' then
       if string.match(opt.netType, "alexnet") then
           tmp_model.modules[22].p = tmp_dr[1]
           tmp_model.modules[19].p = tmp_dr[2]
       elseif string.match(opt.netType, "vgg") then
           tmp_model.modules[35].p = tmp_dr[1]
           tmp_model.modules[38].p = tmp_dr[2]
       end
       tmp_model:add(nn.LogSoftMax())
   elseif tmp_opt=='finetuneM' or tmp_opt=='finetuneR' or tmp_opt=='finetuneMP' or tmp_opt=='finetuneL' or tmp_opt=='trainA' or tmp_opt=='trainL' then
       if string.match(opt.netType, "alexnet") then
           tmp_model.modules[23] = nn.Linear(4096, opt.nClasses)
           tmp_model.modules[23].weight:copy(torch.randn(model.modules[23].weight:size()):mul(0.001)) 
           tmp_model.modules[23].bias:fill(0) -- small uniform numbers
           -- increase dropout rate
           tmp_model.modules[22].p = tmp_dr[1]
           tmp_model.modules[19].p = tmp_dr[2]
       elseif string.match(opt.netType, "vgg") then
           if opt.netType=="vgg_ucfPp5" then
                tmp_model.modules[31]=nn.View(-1,100352)
                tmp_model.modules[32]=nn.Linear(100352, opt.nClasses)
                tmp_model.modules[32].weight:copy(torch.randn(tmp_model.modules[32].weight:size()):mul(0.001)) 
                tmp_model.modules[32].bias:fill(0) -- small uniform numbers
                for j=33,38 do
                    tmp_model.modules[j]=nil
                end
            else
               tmp_model:add(nn.Linear(4096, opt.nClasses))
               tmp_model.modules[39].weight:copy(torch.randn(tmp_model.modules[39].weight:size()):mul(0.001)) 
               tmp_model.modules[39].bias:fill(0) -- small uniform numbers
               -- increase dropout rate
               tmp_model.modules[35].p = tmp_dr[1]
               tmp_model.modules[38].p = tmp_dr[2]
           end
       end
       tmp_model:add(nn.LogSoftMax())
       model:cuda()
   elseif tmp_opt=='finetune' then
       -- tmp_model:insert(nn.Mul(),1);tmp_model.modules[1].weight:fill(255);tmp_model.modules[1]['accGradParameters']=function()end
       tmp_model:add(nn.Log())
   elseif tmp_opt=='truncate' then
       if string.match(opt.netType, "alexnet") then
           tmp_model.modules[23] = nil
           tmp_model.modules[24] = nil
       elseif string.match(opt.netType, "vgg") then
           tmp_model.modules[39] = nil
           tmp_model.modules[40] = nil
           -- increase dropout rate
           tmp_model.modules[35].p = tmp_dr[1]
           tmp_model.modules[38].p = tmp_dr[2]
       end
   end
    collectgarbage()
end


function model_utils.combine_all_parameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--

    -- get parameters
    local networks = {...}
    local parameters = {}
    local gradParameters = {}
    for i = 1, #networks do
        local net_params, net_grads = networks[i]:parameters()

        if net_params then
            for _, p in pairs(net_params) do
                parameters[#parameters + 1] = p
            end
            for _, g in pairs(net_grads) do
                gradParameters[#gradParameters + 1] = g
            end
        end
    end

    local function storageInSet(set, storage)
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
        local Tensor = parameters[1].new

        local storages = {}
        local nParameters = 0
        for k = 1,#parameters do
            local storage = parameters[k]:storage()
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end

        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()

        for k = 1,#parameters do
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage,
                storageOffset + parameters[k]:storageOffset(),
                parameters[k]:size(),
                parameters[k]:stride())
            parameters[k]:zero()
        end

        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset,
                parameters[k]:size(),
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
            local k, v = unpack(storageAndOffset)
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end

        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end




function model_utils.clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

return model_utils

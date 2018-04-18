function createModel(GPUs,numC,numClasses)
   require 'cudnn'
   require 'cunn'

   local modelType = 'D' -- on a titan black, B/D/E run out of memory even for batch-size 32

   -- Create tables describing VGG configurations A, B, D, E
   local cfg = {}
   if modelType == 'A' then
      cfg = {64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'B' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'}
   elseif modelType == 'D' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512}
   elseif modelType == 'E' then
      cfg = {64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'}
   else
      error('Unknown model type: ' .. modelType .. ' | Please specify a modelType A or B or D or E')
   end

   local numF = numC/2;
   local features = nn.Sequential()
   -- hack: multi-stream -> reshape into batch
   features:add(nn.View(-1,numF,224,224))
   do
      local iChannels = numF;
      for k,v in ipairs(cfg) do
         if v == 'M' then
            features:add(cudnn.SpatialMaxPooling(2,2,2,2))
         else
            local oChannels = v;
            local conv3 = cudnn.SpatialConvolution(iChannels,oChannels,3,3,1,1,1,1);
            features:add(conv3)
            features:add(cudnn.ReLU(true))
            features:add(cudnn.SpatialBatchNormalization(oChannels, 1e-3))
            iChannels = oChannels;
         end
      end
   end
   features:cuda()
   if #GPUs > 1 then
      assert(#GPUs <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local features_single = features
      features = nn.DataParallelTable(1)
      for i=1,#GPUs do
         cutorch.setDevice(GPUs[i])
         features:add(features_single:clone():cuda(),i)
      end
   end
   cutorch.setDevice(GPUs[1])
   
   -- remove last maxPool
   local classifier = nn.Sequential()
   classifier:add(nn.View(-1,2*512,14,14))
   classifier:add(cudnn.SpatialConvolution(1024,1024,3,3,1,1,1,1))
   classifier:add(cudnn.ReLU(true))
   classifier:add(cudnn.SpatialBatchNormalization(1024,1e-3))
   classifier:add(nn.Dropout(0.5))
   classifier:add(cudnn.SpatialConvolution(1024,1024,3,3,1,1,1,1))
   classifier:add(cudnn.ReLU(true))
   classifier:add(cudnn.SpatialBatchNormalization(1024,1e-3))
   classifier:add(nn.Dropout(0.5))
   classifier:add(cudnn.SpatialAveragePooling(14,14))
   classifier:add(nn.View(-1,1024))
   classifier:add(nn.Linear(1024,1))
   classifier:add(nn.Sigmoid())
   classifier:cuda()


   local model = nn.Sequential()
   model:add(features):add(classifier)

   return model
end

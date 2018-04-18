require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
require 'image'

local dataset = torch.class('dataLoader')

local initcheck = argcheck{
   pack=true,
   help=[[
     A dataset class for images in a flat folder structure (folder-name is class-name).
     Optimized for extremely large datasets (upwards of 14 million images).
     Tested only on Linux (as it uses command-line linux utilities to scale up)
]],
   {check=function(paths)
       local out = true;
       for k,v in ipairs(paths) do
          if type(v) ~= 'string' then
             print('paths can only be of string input');
             -- out = false
          end
       end
       return out
   end,
    name="paths",
    type="table",
    help="Multiple paths of directories with images"},

   {name="sampleSize",
    type="table",
    help="a consistent sample size to resize the images"},

   {name="split",
    type="number",
    help="Percentage of split to go to Training"
   },

   {name="samplingMode",
    type="string",
    help="Sampling mode: random | balanced ",
    default = "balanced"},

   {name="verbose",
    type="boolean",
    help="Verbose mode during initialization",
    default = false},

   {name="loadSize",
    type="table",
    help="a size to load the images to, initially",
    opt = true},

   {name="forceClasses",
    type="table",
    help="If you want this loader to map certain classes to certain indices, "
       .. "pass a classes table that has {classname : classindex} pairs."
       .. " For example: {3 : 'dog', 5 : 'cat'}"
       .. "This function is very useful when you want two loaders to have the same "
    .. "class indices (trainLoader/testLoader for example)",
    opt = true},

   {name="sampleHookTrain",
    type="function",
    help="applied to sample during training(ex: for lighting jitter). "
       .. "It takes the image path as input",
    opt = true},

   {name="sampleHookTest",
    type="function",
    help="applied to sample during testing",
    opt = true},
}

function dataset:__init(...)

   -- argcheck
   local args =  initcheck(...)
   print(args)
   for k,v in pairs(args) do self[k] = v end

   if not self.loadSize then self.loadSize = self.sampleSize; end

   if not self.sampleHookTrain then self.sampleHookTrain = self.defaultSampleHook end
   if not self.sampleHookTest then self.sampleHookTest = self.defaultSampleHook end

   -- first pass
   -- find class names
   self.classes = {}
   local tnumL=0
   local tnumC=0
   local tmaxPathLength=0
    local file = io.open(self.paths[1])
      if file then
          for line in file:lines() do
              local imgPath, numFrame, classId = unpack(line:split(" ")) 
              tmaxPathLength = math.max(tmaxPathLength,string.len(imgPath))
              --[[
              classId = tonumber(classId)+1
              if tnumC < classId then
                  tnumC=tnumC+1
                  table.insert(self.classes, string.format('%d',classId))
              end
              --]]
              -- for billiard
              classId = tonumber(classId)+1
              if self.classes[classId]==nil then
                  tnumC=tnumC+1
                  table.insert(self.classes, classId)
              end
              tnumL=tnumL+1
          end
      end
   -- find the image path names
   tmaxPathLength = tmaxPathLength+1 -- so that names won't connect
   self.imagePath = torch.CharTensor(tnumL,tmaxPathLength):fill(0)  -- path to each image in dataset
   self.imageClass = torch.Tensor(tnumL) -- class index of each image (class index in self.classes
   self.imageLen = torch.Tensor(tnumL) -- class index of each image (class index in self.classes)
   self.classCount = torch.Tensor(tnumC):fill(0) -- class index of each image (class index in self.classes)
   local s_data = self.imagePath:data()
   -- caffe train/test list: imgPath #frames class label
  local file = io.open(self.paths[1])
  if file then
      local cc=1
      for line in file:lines() do
          local imgPath, numFrame, classId = unpack(line:split(" ")) --unpack turns a table like the one given (if you use the recommended version) into a bunch of separate variables
          classId = tonumber(classId)+1 -- prototxt: 0
          ffi.copy(s_data, imgPath)
          s_data = s_data + tmaxPathLength
          self.imageLen[cc] = tonumber(numFrame);
          self.imageClass[cc] = classId;
          self.classCount[classId]=self.classCount[classId]+1;
          cc=cc+1
      end
  end

   self.classList = {}                  -- index of imageList to each image of a particular class
   self.classListSample = self.classList -- the main list used when sampling data
   self.numSamples = self.imagePath:size(1)
   if self.verbose then print(self.numSamples ..  ' samples found.') end
   --==========================================================================
   print('Updating classList and imageClass appropriately')
   for i=1,tnumC do
      self.classList[i] = torch.Tensor(self.classCount[i])
   end
   local tclassCount = torch.Tensor(#self.classes):fill(1)
    for i=1,tnumL do
        local cid = self.imageClass[i]
        self.classList[cid][tclassCount[cid]] = i
        tclassCount[cid]=tclassCount[cid]+1
   end

   --==========================================================================
   if self.split == 100 then
      self.testIndicesSize = 0
   else
      print('Splitting training and test sets to a ratio of '
               .. self.split .. '/' .. (100-self.split))
      self.classListTrain = {}
      self.classListTest  = {}
      self.classListSample = self.classListTrain
      local totalTestSamples = 0
      -- split the classList into classListTrain and classListTest
      for i=1,#self.classes do
         local list = self.classList[i]
         if list==nil then
            self.classListTrain[i]  = torch.LongTensor()
            self.classListTest[i]  = torch.LongTensor()
         else
             local count = self.classList[i]:size(1)
             local splitidx = math.floor((count * self.split / 100) + 0.5) -- +round
             local perm = torch.randperm(count)
             self.classListTrain[i] = torch.LongTensor(splitidx)
             for j=1,splitidx do
                self.classListTrain[i][j] = list[perm[j]]
             end
             if splitidx == count then -- all samples were allocated to train set
                self.classListTest[i]  = torch.LongTensor()
             else
                self.classListTest[i]  = torch.LongTensor(count-splitidx)
                totalTestSamples = totalTestSamples + self.classListTest[i]:size(1)
                local idx = 1
                for j=splitidx+1,count do
                   self.classListTest[i][idx] = list[perm[j]]
                   idx = idx + 1
                end
             end
         end
      end
      -- Now combine classListTest into a single tensor
      self.testIndices = torch.LongTensor(totalTestSamples)
      self.testIndicesSize = totalTestSamples
      local tdata = self.testIndices:data()
      local tidx = 0
      for i=1,#self.classes do
         local list = self.classListTest[i]
         if list:dim() ~= 0 then
            local ldata = list:data()
            for j=0,list:size(1)-1 do
               tdata[tidx] = ldata[j]
               tidx = tidx + 1
            end
         end
      end
   end
end

-- size(), size(class)
function dataset:size(class, list)
   list = list or self.classList
   if not class then
      return self.numSamples
   elseif type(class) == 'string' then
      return list[self.classIndices[class]]:size(1)
   elseif type(class) == 'number' then
      return list[class]:size(1)
   end
end

-- getByClass
function dataset:getByClass(class,batchId)
   if self.classListSample[class] == nil then
       print('class '..class..' is empty !')
       return nil
   else
       local index = math.ceil(torch.uniform() * self.classListSample[class]:nElement())
       -- assert(index)
       -- print('ss: '..class..' '..index)
       --print(self.classListSample..' '..#self.imagePath)
       local imgpath = ffi.string(torch.data(self.imagePath[self.classListSample[class][index]]))
       local vnF =self.imageLen[self.classListSample[class][index]]
       -- number of frames: start from 1
       local imId = math.ceil(torch.uniform() * (vnF-opt.vnF+1))
       if opt.TcropType[1]==2 then -- middle crop
           imId = math.ceil(0.5*(vnF-opt.vnF+1))
       elseif opt.TcropType[1]==3 then -- use len as temporal crop, vnF=1, start_id (ucf101)
            imId = vnF+math.ceil(torch.uniform() * opt.TcropType[2]-1)
       end
       -- if not paths.filep(imgpath..'/flow_x_0001.jpg') then local dbg = require('debugger');dbg();end
       -- if imId<=0 then local dbg = require('debugger');dbg();end
       return self:sampleHookTrain(imgpath,imId,opt.vnF,batchId,vnF)
   end
end

-- converts a table of samples (and corresponding labels) to a clean tensor
local function tableToOutput(self, dataTable, scalarTable)
   local data, scalarLabels, labels
   local quantity = #scalarTable
   -- assert(dataTable[1]:dim() == 3)
   local nAug=1;
   if opt.testOpt==1 then nAug=10;end -- 10 aug
   scalarLabels = torch.LongTensor(nAug*quantity):fill(-1111)
   local step = 1;
   local numRow=0;
   local p0=0;
   if string.match(torch.type(dataTable[1]),"Tensor") then -- just images
       if string.sub(opt.netType,1,2)=="sm" then -- batchSize vary
           for i =1, quantity do;numRow = numRow+dataTable[i]:size(1);end
           if dataTable[1]:dim() == 2 then
                data = torch.Tensor(numRow,dataTable[1]:size(2))
           end
           for i=1,#dataTable do
              data:narrow(1,1+p0,step):copy(dataTable[i])
              p0 = p0+step
              scalarLabels:narrow(1,1+(i-1)*nAug,nAug):fill(scalarTable[i]+opt.labelOffset) --BCE
              dataTable[i]=nil -- save memory
              scalarTable[i]=nil
           end
       else
           if opt.retrainOpt=='test' then
               step = dataTable[1]:size(1)
           end
           if dataTable[1]:dim() == 4 then -- siamese
               data = torch.Tensor(quantity*step,
                       dataTable[1]:size(2), dataTable[1]:size(3), dataTable[1]:size(4))
           elseif dataTable[1]:dim() == 3 then 
               data = torch.Tensor(quantity*step,
                       self.sampleSize[1], self.sampleSize[2], self.sampleSize[3])
           elseif dataTable[1]:dim() == 2 then
                   data = torch.Tensor(quantity*step,
                       dataTable[1]:size(1), dataTable[1]:size(2))
           else
               data = torch.Tensor(quantity*step,dataTable[1]:size(1))
           end
          for i=1,#dataTable do
              data:narrow(1,(i-1)*step+1,step):copy(dataTable[i])
              scalarLabels:narrow(1,1+(i-1)*nAug,nAug):fill(scalarTable[i]+opt.labelOffset) --BCE
              dataTable[i]=nil -- save memory
              scalarTable[i]=nil
           end
       end
       -- local dbg = require('debugger');dbg()
   else --image+bbox
       local numB = 0
       for i=1,#dataTable do numB=numB+dataTable[i][2]:size(1) end
       data = {torch.Tensor(quantity,dataTable[1][1]:size(1), opt.sampleSize[1], opt.sampleSize[2]),
                torch.Tensor(numB,5)}
       if #dataTable==1 then
          data[1]:copy(dataTable[1][1])
          data[2]:copy(dataTable[1][2])
          scalarLabels[1]=scalarTable[1]+opt.labelOffset
       else
           numB=1
           local numB2
           for i=1,#dataTable do
              data[1][i]:copy(dataTable[i][1])
              numB2 = dataTable[i][2]:size(1)
              data[2]:narrow(1,numB,numB2):copy(dataTable[i][2])
              numB=numB+numB2
              scalarLabels[i] = scalarTable[i]+opt.labelOffset
              dataTable[i]=nil
              scalarTable[i]=nil
           end
        end
        dataTable=nil
        scalarTable=nil
   end
  collectgarbage();collectgarbage();
   return data, scalarLabels
end

-- sampler, samples from the training set.
function dataset:sample(quantity)
   assert(quantity>0)
   local dataTable = {}
   local scalarTable = {}
   if opt.batchRand==0 then -- purely random
       for i=1,quantity do
          local out =nil
          local classId
          while out==nil do
              classId = torch.random(1, #self.classes)
              out = self:getByClass(classId,i)
          end
          table.insert(dataTable, out)
          table.insert(scalarTable, classId)
       end
   elseif opt.batchRand==1 then -- proportional to class size
        local classId=torch.multinomial(self.classCount,quantity,true);
        for i=1,quantity do
          local out = self:getByClass(classId[i],i)
          table.insert(dataTable, out)
          table.insert(scalarTable, classId[i])
       end
   end
   local data, scalarLabels = tableToOutput(self, dataTable, scalarTable)
   return data, scalarLabels
end

function dataset:get(i1, i2)
   local indices = torch.range(i1, i2);
   local quantity = i2 - i1 + 1;
   assert(quantity > 0)
   -- now that indices has been initialized, get the samples
   local dataTable = {}
   local scalarTable = {}
   local vid
   for i=1,quantity do
      -- load the sample
      local index = indices[i]
      local imgpath = ffi.string(torch.data(self.imagePath[index]))
      local vnF =self.imageLen[index]
      -- e.g 0-9/10
      if opt.TcropType[1]==1 then
        -- default: test range
        vid = 1+math.ceil( opt.testRange[2]*(vnF-opt.vnF)/(opt.testRange[1]-1) )
        if string.match(imgpath,"bflowb") then
          -- same frame in the backward order: end -> start of the sample
          vid = (vnF+1)-(vid+opt.vnF-1);
        end
      else -- vnF:start_id
        vid = vnF+opt.TcropType[2]
      end

      -- print(i,imgpath,vid,opt.vnF,vnF)
      local out = self:sampleHookTest(imgpath,vid,opt.vnF,i,vnF)
      table.insert(dataTable, out)
      -- local dbg = require('debugger');dbg()
      table.insert(scalarTable, self.imageClass[index])
   end
   -- local dbg = require('debugger');dbg()
   local data, scalarLabels = tableToOutput(self, dataTable, scalarTable)
   -- local dbg = require('debugger');dbg()
   return data, scalarLabels
end

return dataset

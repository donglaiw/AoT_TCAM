--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'image'
paths.dofile('dataset_video.lua')
paths.dofile('util/misc.lua')

matio = require 'matio'
-- image.crop: 0-index
-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainCache = paths.concat(opt.cache, 'trainCache.t7')
local testCache = paths.concat(opt.cache, opt.testSet..'Cache.t7')
local meanstdCache = paths.concat(opt.cache, 'meanstdCache.t7')

-- Check for existence of opt.data
--[[
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end
--]]
local function inputCaffeFlow(out)
    -- vgg style
    out = out*255; 
    out = out:add(-128) -- may have some roundoff error
    -- out = out:permute(1,3,2):add(-128)
   return out
end
local function cornerCrop(cid,iW,iH,oW,oH)
    if cid==1 then
        return {0,0}
    elseif cid==2 then
        return {0,iW-oW}
    elseif cid==3 then
        return {iH-oH,0}
    elseif cid==4 then
        return {iH-oH,iW-oW}
    elseif cid==5 then
        return {math.ceil((iH-oH)/2),math.ceil((iW-oW)/2)}
    -- other four patches    
    elseif cid==6 then
        return {0,math.ceil((iW-oW)/2)}
    elseif cid==7 then
        return {math.ceil((iH-oH)/2),0}
    elseif cid==8 then
        return {iH-oH,math.ceil((iW-oW)/2)}
    elseif cid==9 then
        return {math.ceil((iH-oH)/2),iW-oW}
    end
end
local function inputCaffe(out)
    -- vgg style
    local tmp_mean={103.939, 116.779, 123.68}
    local tmp;
    -- rgb -> bgr
    tmp=out:clone();out[3]=tmp[1];out[1]=tmp[3]
    out=out*255; 
    -- no need to permute: see tests/T_loadVGG.lua
    -- out = out:permute(1,3,2)
   for i=1,3 do -- channels
      out[{{i},{},{}}]:add(-tmp_mean[i])
   end
   return out
end

local function loadRoi(path,imId,L,batchId)
    local bboxN
    local fn = strSplit(path,'/')
    local fn2="f"
    local fid=imId+math.ceil(L/2)
    if opt.bType==0 then -- billiard 
        local tmp = strSplit(fn[#fn-3],'_')
        fn[#fn-3]=tmp[1]..'_pos'
        fn[#fn]=fn[#fn]..opt.bTypeSuf
        fn2 = string.sub(tmp[2],1,1)
        bboxN = '/'..table.concat(fn,'/')..opt.bTypeFolder..'.mat'
    elseif opt.bType==0.1 then -- 3c
        fn2 = fn[#fn-2][9]
        fn[#fn-2]='3c_trajR'
        bboxN = '/'..table.concat(fn,'/')..'.mat'
    elseif opt.bType==1 then --lip-face 
        bboxN ='/data/vision/billf/deep-time/Data/talk_AZ/data/bboxFa.mat'
        -- fixed/same position any way
        -- fn2 = string.sub(fn[#fn-3],1,1)
    elseif opt.bType==1.1 then --lip-mouth 
        bboxN ='/data/vision/billf/deep-time/Data/talk_AZ/data/bboxMo.mat'
        -- fn2 = string.sub(fn[#fn-3],1,1)
    elseif opt.bType==2 then -- ucf/mpii 
        fn[#fn-2]=opt.bTypeFolder
        fn2 = fn[#fn-2][1]
        bboxN = '/'..table.concat(fn,'/')..'.mat'
    end
    if opt.lstmN>=1 then
        fid = imId+torch.range(0,opt.lstmN-1)*(L/opt.lstmN)+math.ceil(L/opt.lstmN/2)
    end

    local output;
    if L==1 then -- image
        -- look it up
    else --video
        local bbox = matio.load(bboxN,'bbox')

         if fn2=='b' then
            -- flip the bbox order
            fid = #bbox+1-fid
            -- [wrong] make sure the center has the same relative pos
            -- if L%2 == 0 then; fid=fid+1; end
        end
        if opt.bType==1.1 then 
            fid=1
        end
        if opt.bloadType==0 then -- all boxes
            output = torch.FloatTensor(bbox[fid]:size(1),5):fill(batchId)
            output:narrow(2,2,4):copy(bbox[fid]:narrow(2,1,4))
        elseif opt.bloadType==1 then -- +whole
            if bbox==nil or bbox[fid]==nil then
                output = torch.FloatTensor(1,5):fill(batchId)
            else
                output = torch.FloatTensor(bbox[fid]:size(1)+1,5):fill(batchId)
                output:narrow(1,1,bbox[fid]:size(1)):narrow(2,2,4):copy(bbox[fid]:narrow(2,1,4))
            end
            output[-1][2]=1;output[-1][3]=1;output[-1][4]=1000;output[-1][5]=1000;
        elseif opt.bloadType==2 then -- person (or whole if none)
            local noBox=true
            if bbox~=nil and bbox[fid]~=nil then
                local indices  = torch.range(1, bbox[fid]:size(1)):long();
                indices = indices[bbox[fid]:narrow(2,5,1):eq(0)]
                if #indices~=0 then
                    bbox[fid] = bbox[fid]:index(1,indices)
                    noBox=false
                    output = torch.FloatTensor(bbox[fid]:size(1),5):fill(batchId)
                    output:narrow(2,2,4):copy(bbox[fid]:narrow(2,1,4))
                end
            end
            if noBox then
                output = torch.FloatTensor{batchId,1,1,1000,1000}
            end
        elseif opt.bloadType==3 then -- load bbox from multiple time-stamp
            output = torch.FloatTensor(fid:size(1),5)
            -- random select one box: consistent over time
            local boxId = math.floor(torch.uniform(1,bbox[fid[1]]:size(1)+1))
            for i=1,fid:size(1) do
                output[i][1] = (batchId-1)*opt.lstmN+i
                output:narrow(1,i,1):narrow(2,2,4):copy(bbox[fid[i]]:narrow(1,boxId,1):narrow(2,1,4))
            end
        elseif opt.bloadType==4 then -- load max area bbox
            if bbox[fid]:size(1)==1 then
                output = bbox[fid]:clone()
            else
                -- random select one box: consistent over time
                local area = torch.Tensor(bbox[fid]:size(1)-1):fill(0)
                for i=2,bbox[fid]:size(1) do
                    tmp = bbox[fid]:narrow(1,i,1)
                    area[i-1]=(tmp[1][3]-tmp[1][1]+1)*(tmp[1][4]-tmp[1][2]+1)
                end
                _,mid = torch.max(area,1)
                output=bbox[fid]:narrow(1,1+mid[1],1):clone()

            end
        end
    end
    return output
end

local function loadImage(path,imId,L)
   -- print('load: ' .. path)
   if opt.xType==3 then
       L=L+1; -- rgb diff
   end
   local step = 1
   local sufType = 'jpg'
   local suf={''}
    if opt.imType==2 then
        suf={'_x','_y'}
    elseif opt.imType==3 then
        suf={'_x'}
    end
    -- hack rgb input without flip images with soft-link
    -- print(path,opt.imType,imId)
    if opt.imType==1 and string.match(path,'hm20_flow') then
        sufType = 'png'
        local index = string.find(string.sub(path,1,-4), "/[^/]*$");
        imId = imId+tonumber(string.sub(path,index+1,-4))-1
        if string.match(path,'/b/') then -- reverse order
            step = -1
            imId = imId+L+1
        end
        path = string.sub(path,1,-4) .. '/im/'
    end
   local inputs = nil
    for j=1,#suf do
        for i =1,L do    
           -- if not paths.filep(path..string.format(opt.imFormat,suf[j],imId+i-1)) then local dbg = require('debugger');dbg() end
           local input = image.load(path..'/'..string.format(opt.imFormat,suf[j],imId+i*step-1,sufType))
           -- print(path..'/'..string.format(opt.imFormat,suf[j],imId+i*step-1,sufType))
           if input:dim() == 2 then -- 1-channel image loaded as 2D tensor
              input = input:view(1,input:size(1), input:size(2)):repeatTensor(3,1,1)
           elseif (opt.imType~=2 and opt.imType~=3) and input:dim() == 3 and input:size(1) == 1 then -- 1-channel image
              input = input:repeatTensor(3,1,1)
           elseif input:dim() == 3 and input:size(1) == 4 then -- image with alpha
              input = input[{{1,3},{},{}}]
           end
           -- old FRCNN: crop the flow directly
           if (not string.match(opt.netType,'RCNN')) and opt.bType~=0 then -- quick hack
               if opt.bType==1.1 then -- lip mouth
                   input=image.crop(input,76,106,173,203)
                   -- input=image.crop(input,106,76,203,173)
               end
           end

           -- find the smaller dimension, and resize it to 256 (while keeping aspect ratio)
           local iC = input:size(1)
           local iH = input:size(2)
           local iW = input:size(3)
           -- loadSize: H-W
           -- print('bb:');print(input:size())
           -- keep aspect ratio, short side match
           -- scale: width-height
           -- short-end size
           if iW < iH then --tall
               if iW~= opt.loadSize[2] then
                  input = image.scale(input, opt.loadSize[2], opt.loadSize[2] * iH / iW)
              end
           else --long
               if iH~= opt.loadSize[1] then
                  input = image.scale(input, opt.loadSize[1] * iW / iH, opt.loadSize[1])
              end
           end
           if inputs==nil then
               inputs=torch.FloatTensor(L*iC*#suf,input:size(2),input:size(3)) 
           end
           -- shuffle the order xxxx,yyyy
           -- inputs:narrow(1,(i-1)*(iC)+(j-1)*iC*L+1,iC):copy(input)
           -- shuffle the order xyxyxy
           inputs:narrow(1,(i-1)*(iC*#suf)+(j-1)*iC+1,iC):copy(input)
       end
   end
   return inputs
end


-- channel-wise mean and std. Calculate or load them from disk later in the script.
local mean,std
--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
--]]
-- function to process the image
local imPreprocess = function(out,path,imId,L,batchId,hflip)
   -- remove mean/std
    if opt.xType==3 then -- d-rgb
        out = 255*torch.add(out:sub(1,out:size(1)-opt.vnC),-out:sub(opt.vnC+1,out:size(1)))
    elseif opt.xType==4 then
       out = inputCaffe(out)
    elseif opt.xType==5 then
       out = inputCaffeFlow(out)
    elseif opt.xType==6 then
       out = torch.abs(inputCaffeFlow(out))
    else
       for i=1,3 do -- channels
          if mean then out[{{i},{},{}}]:add(-mean[i]) end
          if std then out[{{i},{},{}}]:div(std[i]) end
       end
    end

    -- flip image
    if hflip==1 then 
        out = image.hflip(out)
        -- flip flow
        if (opt.xType>=5 and opt.xType~=6 and opt.xType<=8) then
            if opt.imType==3 then -- only fx input
                out:mul(-1)
            elseif opt.imType==2 then --fx+fy input
                for i=1,out:size(1),2 do
                    out:narrow(1,i,1):mul(-1)
                end
            end
        end
    end
 
    -- apply the box
    if opt.xType>=7 and opt.xType<=8 then
        -- select the box
       --print(path,imId)
       local bbox = loadRoi(path,imId,L,batchId)
        -- quick hack: bbox: imresize round off differently
        bbox[{{},{3}}]:cmin(iW);
        bbox[{{},{4}}]:cmin(iH)
        if hflip==1 then
            local tmp = bbox:clone()
            bbox[{{},{1}}] = iW+1-tmp[{{},{3}}]
            bbox[{{},{3}}] = iW+1-tmp[{{},{1}}]
        end
        if opt.xType>=7 and opt.xType<=8 then
            -- bbox -> square
            bbox=bbox[1]
            local b0=bbox:clone()
            local bW=bbox[3]-bbox[1]+1
            local bH=bbox[4]-bbox[2]+1
            local bDif = torch.abs(bH-bW)
            if bDif~=0 then
                local bd=torch.Tensor(2);
                if bW>bH then
                    bd[1]=bbox[2]-1;bd[2]=iH-bbox[4];
                    if torch.min(bd)>bDif then
                        -- even distribute
                        bbox[2]=bbox[2]-torch.floor(bDif/2)
                        bbox[4]=bbox[4]+torch.ceil(bDif/2)
                    else
                        if bd[1]<bd[2] then
                            bbox[2]=1;bbox[4]=bH
                        else
                            bbox[2]=iH-bH+1;bbox[4]=iH
                        end
                    end
                else
                    bd[1]=bbox[1]-1;bd[2]=iW-bbox[3];
                    if torch.min(bd)>bDif then
                        -- even distribute
                        bbox[1]=bbox[1]-torch.floor(bDif/2)
                        bbox[3]=bbox[3]+torch.ceil(bDif/2)
                    else
                        if bd[1]<bd[2] then
                            bbox[1]=1;bbox[3]=bW
                        else
                            bbox[1]=iW-bH+1;bbox[3]=iW
                        end
                    end
                end
            end
            -- crop the flow value
            if bbox[1]<1 or bbox[2]<1 or bbox[3]>iW or bbox[4]>iH then
                print(path,imId)
                print(b0)
                print(bbox)
                print(out:size())
            end
           out = image.scale(image.crop(out, bbox[1], bbox[2], bbox[3], bbox[4]),opt.sampleSize[2], opt.sampleSize[2])
           if opt.xType==8 then
                out:mul(opt.sampleSize[2]/(bbox[4]-bbox[2]+1));
           end
        end
   end
    if string.match(opt.netType,'RCNN') then
       -- image + bbox
       local bbox = loadRoi(path,imId,L,batchId)
        if hflip==1 then
            local tmp = bbox:clone()
            -- switch [2,4] and width
            bbox[{{},{2}}] = iW+1-tmp[{{},{4}}]
            bbox[{{},{4}}] = iW+1-tmp[{{},{2}}]
        end
        -- if box sample size is different from image size
        -- if opt.loadSize[1]~=256 then -- suppose
        -- end
        if opt.sampleSize[1]~=-1 then -- f-rcnn: feed-in and interp p5
            -- cropped bbox
            bbox[{{},{3}}]:add(-h1);bbox[{{},{5}}]:add(-h1)
            bbox[{{},{2}}]:add(-w1);bbox[{{},{4}}]:add(-w1)
            -- bbox range
            bbox:narrow(2,2,4):cmax(1)
            bbox[{{},{3}}]:cmin(sampleSz[1]);bbox[{{},{5}}]:cmin(sampleSz[1])
            bbox[{{},{2}}]:cmin(sampleSz[2]);bbox[{{},{4}}]:cmin(sampleSz[2])
        end
        -- at least one box, the overall
        local toDel = (bbox[{{},{3}}]-bbox[{{},{5}}]):eq(0) + (bbox[{{},{2}}]-bbox[{{},{4}}]):eq(0)
        local indices  = torch.range(1, bbox:size(1)):long();
        indices = indices[toDel:eq(0)]
        if indices:dim(1)==0 then -- fall back, if no intersect
            bbox = bbox:index(1, indices)
            bbox = torch.FloatTensor{bbox[1][1],1,1,sampleSz[2],sampleSz[1]}
        else
            if opt.bNum==1 then -- sample a box (no loss)
               indices =  indices:narrow(1,math.floor(torch.uniform(1,indices:size(1)+1)),1)
            end
            -- indices[1]=1 -- debug: should be the same as CNN
            bbox = bbox:index(1, indices)
        end
        out = {out,bbox}
   end
   return out
end
local function cropImage(input,imId,cropId)
   local out;
   -- crop
   local iH = input:size(2)
   local iW = input:size(3)
   -- self.sampleSize: from cache, not flexible to change
   local oH = opt.sampleSize[1]
   local oW = opt.sampleSize[2]
   -- if iH=oH -> need h1=0 [image.crop updated]
   local w1=0
   local h1=0
   local sampleSz={oH,oW}
   if sampleSz[1]==-1 then
        sampleSz[1]=iH; sampleSz[2]=iW;
   end
   if opt.cropType==0 or oH==-1 or (oH==iH and oW==iW) then
       out = input
   else
       if opt.cropType==1 then
           -- do random crop
           h1 = math.ceil(torch.uniform(0, iH-oH))
           w1 = math.ceil(torch.uniform(0, iW-oW))
       elseif opt.cropType==2 then
           -- do fixed 5-crop: for training
           local pos = cornerCrop(math.ceil(torch.uniform(0, 5)),iW,iH,oW,oH)
           --local pos= cropOpt[math.ceil(torch.uniform(0, 5)]
           w1=pos[2];h1=pos[1]
       elseif opt.cropType==2.1 then
           -- one of fixed 5-crop: for testing
           local pos = cornerCrop(cropId,iW,iH,oW,oH)
           w1=pos[2];h1=pos[1]
       elseif opt.cropType==3 then
           -- do fixed 3-crop (vertical=256)
           local ind={1,2,6};
           local pos = cornerCrop(ind[math.ceil(torch.uniform(0, 3))],iW,iH,oW,oH)
           --local pos= cropOpt[math.ceil(torch.uniform(0, 5)]
           w1=pos[2];h1=pos[1]
       elseif opt.cropType==4 then -- for 1-test
           w1 = math.ceil((iW-oW)/2)
           h1 = math.ceil((iH-oH)/2)
       end
       out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   end
   return out;
end
-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, path, imId, L, batchId, nF)
   collectgarbage()
  -- single video
   local out;
   if opt.dType[1]=='mat' then
       out = matio.load(path..((imId-1)*opt.dNum+1)..'.mat',opt.dType[2])
       if opt.netType=='smGAP' then
           out=torch.reshape(out,14,14,1024)
           out=torch.reshape(out:transpose(1,3),1,opt.vnC)
       end
       -- out = torch.reshape(matio.load(path..((imId-1)*opt.dNum+1)..'.mat',opt.dType[2]),opt.vnC)
   else
       local hflip = 0
       if torch.uniform() > 0.5 then hflip=1;end
       if string.match(path,',') then --siamese
           local paths=strSplit(path,',')
           local imId_b = (nF+1)-(imId+L-1)
           local input1 = loadImage(paths[1],imId,L)
           local input2 = loadImage(paths[2],imId_b,L)
           out = torch.cat(
              torch.reshape(imPreprocess(cropImage(input1,imId),paths[1],imId,L,batchId,hflip),1,self.sampleSize[1],self.sampleSize[2],self.sampleSize[3]), 
              torch.reshape(imPreprocess(cropImage(input2,imId),paths[2],imId_b,L,batchId,hflip),1,self.sampleSize[1],self.sampleSize[2],self.sampleSize[3])
            ,1)
       else
           local input1 = loadImage(path,imId,L)
           hflip=0
           out = imPreprocess(cropImage(input1,imId,L),path,imId,batchId,hflip)
            -- local matio = require 'matio'; matio.save('db.mat',{a=input1,b=cropImage(input1,imId),c=out});local dbg = require('debugger');dbg()
           -- print(path,imId,hflip)
       end
   end
   collectgarbage();
   return out
end

if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   --[[
   assert(trainLoader.paths[1] == opt.data:gsub('@', 'train'),
          'cached files dont have the same path as opt.data. Remove your cached files at: '
             .. trainCache .. ' and rerun the program')
   --]]
else
   print('Creating train metadata')
   -- for dataset_video: xxx/train
   local tmp = opt.data:gsub('@', 'train')
   trainLoader = dataLoader{
      paths = {tmp},
      loadSize = {opt.nC, opt.loadSize[1], opt.loadSize[2]},
      sampleSize = {opt.nC, opt.sampleSize[1], opt.sampleSize[2]},
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end

collectgarbage()

-- do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")

end

-- End of train loader section
--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns an image's
--]]

local testHook = function(self, path, imId, L, batchId,nF)
   collectgarbage()
   local out,cfg,tmp;
   local paths={path};
   if opt.dType[1]=='mat' then
       -- in batch
       if string.sub(path,#path-3,#path) == ".mat" then
           tmp = matio.load(path,opt.dType[2])
       else
           tmp = matio.load(path..((imId-1)*opt.dNum+1)..'.mat',opt.dType[2])
       end

       -- matlab reshape (train) and torch (test) is different !!!
       local num=tmp:nElement()/tmp:size(1)
       -- print(tmp:size())
       if opt.netType=='smGAP' then -- torch reshape
            tmp = torch.reshape(tmp,tmp:size(1),num)
       else -- same with train: matlab reshape
            tmp = torch.reshape(tmp:transpose(2,4),tmp:size(1),num)
       end
       -- tmp = torch.reshape(tmp,tmp:size(1),tmp:nElement()/tmp:size(1))
       if opt.testOpt==0 then -- only test the center
           out=torch.zeros(tmp:size(1)/10,num)
           -- out=tmp:narrow(1,1,tmp:size(1)/10)
           for i=1,tmp:size(1)/10 do
               out:narrow(1,i,1):copy(tmp:narrow(1,5+(i-1)*10,1))
               -- out:narrow(1,i,1):copy(torch.reshape(tmp:narrow(1,5+(i-1)*10,1):transpose(2,4),1,num))
           end
       elseif opt.testOpt==1 then -- all 10 version
           -- need to fix
           out=tmp
           --[[
           -- no reshape of tmp
           if tmp:dim()==2 then
               out=tmp
           else
               out=torch.zeros(tmp:size(1),num)
               for i=1,tmp:size(1) do
                   out:narrow(1,i,1):copy(torch.reshape(tmp:narrow(1,i,1):transpose(2,4),1,num))
               end
           end
               --]]
       end
   else
       if opt.testOpt==0 then -- only test the center
           cfg=torch.Tensor(2,1):fill(5); -- cropType will do (cropType=2.1)
       elseif opt.testOpt==1 then -- all 10 version
           cfg=torch.Tensor(2,10):fill(0);
           -- row 1: 0-5,0-5
           -- row 2: 0-0,1-1
           cfg:narrow(1,1,1):narrow(2,1,5):copy(torch.range(1,5))
           cfg:narrow(1,1,1):narrow(2,6,5):copy(torch.range(1,5))
           cfg:narrow(1,2,1):narrow(2,6,5):fill(1)
       elseif opt.testOpt==2 then -- all 10 version
           cfg=torch.Tensor(2,8):fill(0);
           -- row 1: 0-5,0-5
           -- row 2: 0-0,1-1
           cfg:narrow(1,1,1):narrow(2,1,4):copy(torch.range(6,9))
           cfg:narrow(1,1,1):narrow(2,5,4):copy(torch.range(6,9))
           cfg:narrow(1,2,1):narrow(2,5,4):fill(1)
       end
       if string.match(path,',') then --siamese
           paths=strSplit(path,',')
       end
       local step=#paths; 
       if #paths==2 then -- fb pair
           local imId_b = (nF+1)-(imId+L-1)
           local input1 = loadImage(paths[1],imId,L)
           local input2 = loadImage(paths[2],imId_b,L)

           for i=1,cfg:size(2) do
               local out1=cropImage(input1,imId,L,cfg[1][i])
               local out2=cropImage(input2,imId,L,cfg[1][i])
               if i==1 then
                   out=torch.Tensor(cfg:size(2),out1:size(1)*2,out1:size(2),out1:size(3))
               end
                out:narrow(1,step*(i-1)+1,step):copy(torch.cat(1,imPreprocess(out1,paths[1],imId,L,batchId,cfg[2][i]),imPreprocess(out2,paths[2],imId_b,L,batchId,cfg[2][i])))
            end
       else
           local input1 = loadImage(paths[1],imId,L)
           for i=1,cfg:size(2) do
                local out1=cropImage(input1,imId,cfg[1][i])
                if i==1 then
                    out=torch.Tensor(cfg:size(2),out1:size(1),out1:size(2),out1:size(3))
                end
               out:narrow(1,step*(i-1)+1,step):copy(imPreprocess(out1,paths[1],imId,L,batchId,cfg[2][i]))
           end
       end
    end
   -- local matio = require 'matio'; matio.save('db.mat',{a=out});local dbg = require('debugger');dbg()
   -- local dbg = require('debugger');dbg()
   -- p out:narrow(1,1,1):narrow(2,1,1)-out:narrow(1,2,1):narrow(2,1,1)
   return out
end

if paths.filep(testCache) then
   print('Loading test metadata from cache: '..testCache)
   testLoader = torch.load(testCache)
   testLoader.sampleHookTest = testHook
   --[[
   assert(testLoader.paths[1] == opt.data:gsub('@', opt.testSet),
          'cached files dont have the same path as opt.data. Remove your cached files at: '
             .. testCache .. ' and rerun the program')
   --]]
else
   print('Creating test metadata')
   local tmp = opt.data:gsub('@', opt.testSet)
   testLoader = dataLoader{
      paths = {tmp},
      loadSize = {opt.nC, opt.loadSize[1], opt.loadSize[2]},
      sampleSize = {opt.nC, opt.sampleSize[1], opt.sampleSize[2]},
      split = 100,
      verbose = true,
      forceClasses = trainLoader.classes -- force consistent class indices between trainLoader and testLoader
   }
   torch.save(testCache, testLoader)
   testLoader.sampleHookTest = testHook
end
collectgarbage()
-- End of test loader section

-- Estimate the per-channel mean/std (so that the loaders can normalize appropriately)
if opt.doMeanStd==1 then
    if  paths.filep(meanstdCache) then
       local meanstd = torch.load(meanstdCache)
       mean = meanstd.mean
       std = meanstd.std
       print('Loaded mean and std from cache.')
    else
       local tm = torch.Timer()
       local nSamples = 1000
       print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
       local meanEstimate = {0,0,0}
       for i=1,nSamples do
          local img = trainLoader:sample(1)[1]
          if img:dim()==3 then
              for j=1,3 do
                 meanEstimate[j] = meanEstimate[j] + img[j]:mean()
              end
          else
              meanEstimate[1] = meanEstimate[1] + img:mean()
          end
       end
       for j=1,3 do
          meanEstimate[j] = meanEstimate[j] / nSamples
       end
       mean = meanEstimate

       print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
       local stdEstimate = {0,0,0}
       for i=1,nSamples do
          local img = trainLoader:sample(1)[1]
          if img:dim()==3 then
              for j=1,3 do
                 stdEstimate[j] = stdEstimate[j] + img[j]:std()
              end
          else
            stdEstimate[1] = stdEstimate[1] + img:std()
          end
       end
       for j=1,3 do
          stdEstimate[j] = stdEstimate[j] / nSamples
       end
       std = stdEstimate

       local cache = {}
       cache.mean = mean
       cache.std = std
       torch.save(meanstdCache, cache)
       print('Time to estimate:', tm:time().real)
    end
else
    -- hack the flow [mean and std]
    local cache = {}
    if opt.imType==2 or opt.imType==3 then      
       local img = torch.Tensor(trainLoader.sampleSize)
       cache.mean = img:fill(0.5)
       cache.std = nil
   end
       torch.save(meanstdCache, cache)

end

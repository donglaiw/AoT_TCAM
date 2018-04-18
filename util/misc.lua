
-- misc utilities
function gVars()
    for x,v in pairs(upvalues()) do print(x,v) end
end
function upvalues()
  local variables = {}
  local idx = 1
  local func = debug.getinfo(2, "f").func
  while true do
    local ln, lv = debug.getupvalue(func, idx)
    if ln ~= nil then
      variables[ln] = lv
    else
      break
    end
    idx = 1 + idx
  end
  return variables
end

function strSplit(tmpStr,delim)
  local param = {}
   local tmp_c=1;
   for tmp_d in string.gmatch(tmpStr,"([^"..delim.."]+)") do
       param[tmp_c] = tmp_d;
       tmp_c = tmp_c + 1
   end
   return param
end
-- can reverse table
function sliceTable(input,i1,i2,step)
    local output={}
    for i=i1,i2,step do
        table.insert(output,input[i])
    end
    return output
end

function clone_list(tensor_list, zero_too)
    -- utility function. todo: move away to some utils file?
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end
-- shuffle class label
function vggLabel(tmpLoader)
    local file='/data/vision/billf/donglai-lib/VisionLib/Donglai/DeepL/caffe/data/ilsvrc12/synsets.txt'
    local vggL={}
    for line in io.lines(file) do 
        table.insert(vggL,line)
    end
    local dL={}
    for k,v in pairs(tmpLoader.classes) do
        dL[v]=k
    end
    -- start to change
    local runningIndex = 0
    local tmpList={}
    for i=1,#tmpLoader.classes do
        tmpList[i] = tmpLoader.classList[i]:clone()
    end
    for i=1,#tmpLoader.classes do
      tmpLoader.classList[i] = tmpList[dL[vggL[i]]]:clone()
      tmpLoader.imageClass[{{tmpLoader.classList[i][1], tmpLoader.classList[i][-1]}}]:fill(i)
    end
    for i=1,#tmpLoader.classes do
        tmpLoader.classes[i]=vggL[i]
        tmpLoader.classCount[i] = tmpLoader.classList[i]:size(1)
   end
end



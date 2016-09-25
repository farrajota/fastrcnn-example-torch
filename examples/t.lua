require 'nn'
require 'cudnn'

local function ConvertBNcudnn2nn(net)
  
  local function ConvertModule(net)
    return net:replace(function(x)
        if torch.type(x) == 'cudnn.BatchNormalization' then
          return cudnn.convert(x, nn)
        else
          return x
        end
    end)
  end
  
  net:apply(function(x) return ConvertModule(x) end)
  
end


local net = nn.Sequential()
net:add(nn.Linear(10,100))
net:add(cudnn.BatchNormalization(100))
net:add(nn.Sequential():add(cudnn.BatchNormalization(100)))
print(net)


ConvertBNcudnn2nn(net)
print(net)
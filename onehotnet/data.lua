--[[
Data class for OnehotNet
Copyright 2016 Xiang Zhang
--]]

local class = require('pl.class')
local torch = require('torch')

local parent = require('glyphnet/data')

local Data = class(parent)

-- Constructor for Data
-- config: configuration table
--   .file: file for data
--   .batch: batch of data
--   .size: size of the quantization
function Data:_init(config)
   local data = torch.load(config.file)
   self.data = {code = data.index, code_value = data.content}
   self.length = config.length or 2048
   self.size = config.size or 256
   self.batch = config.batch or 16
end

function Data:initSample(sample, label)
   local sample = sample or torch.Tensor(self.batch, self.size, self.length)
   local label = label or torch.Tensor(self.batch)
   sample:zero()
   return sample, label
end

function Data:index(sample, class, item)
   local code, code_value = self.data.code, self.data.code_value
   local position = 1

   for field = 1, code[class][item]:size(1) do
      -- Break if current position is larger than sample length
      if position > sample:size(2) then
         break
      end
      for char = 1, code[class][item][field][2] + 1 do
         -- Break if current position is larger than sample length
         if position > sample:size(2) then
            break
         end
         local char_index = code[class][item][field][1] + char - 1
         sample[code_value[char_index] + 1][position] = 1
         position = position + 1
      end
   end

   return sample
end

return Data

--[[
Data class for Embedding Net
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
--   .replace: the code to for replacing padding space
function Data:_init(config)
   self.data = torch.load(config.file)
   self.length = config.length or 512
   self.batch = config.batch or 16
   self.replace = config.replace or 65537
   self.shift = config.shift or 0
end

function Data:initSample(sample, label)
   local sample = sample or torch.Tensor(self.batch, self.length)
   local label = label or torch.Tensor(self.batch)
   sample:fill(self.replace)
   return sample, label
end

function Data:index(sample, class, item)
   local code, code_value = self.data.code, self.data.code_value
   local position = 1

   for field = 1, code[class][item]:size(1) do
      -- Break if current position is larger than sample length
      if position > sample:size(1) then
         break
      end
      -- Determine the actual length
      local length = code[class][item][field][2]
      if position + length - 1 > sample:size(1) then
         length = sample:size(1) - position + 1
      end
      -- Copy the data over
      if length > 0 then
         sample:narrow(1, position, length):copy(
            code_value:narrow(1, code[class][item][field][1], length)):add(
            self.shift)
      end
      -- Increment the position value
      position = position + length
   end

   return sample
end

return Data

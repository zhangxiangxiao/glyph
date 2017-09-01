--[[
Data class for LinearNet
Copyright 2016 Xiang Zhang
--]]

local class = require('pl.class')
local math = require('math')
local torch = require('torch')

local Data = class()

-- Constructor for Data
-- config: configuration table
--   .file: the data file location
-- data_table: if present, will use the data_table instead of load from file
function Data:_init(config, data_table)
   self.data = data_table or torch.load(config.file)
end

function Data:getClasses()
   return #self.data.bag
end

function Data:getSample(sample, label)
   local bag, bag_index, bag_value =
      self.data.bag, self.data.bag_index, self.data.bag_value

   -- Sample a non-empty example
   local class = torch.random(#bag)
   local item = torch.random(bag[class]:size(1))
   while bag[class][item][2] == 0 do
      class = torch.random(#bag)
      item = torch.random(bag[class]:size(1))
   end

   local start = bag[class][item][1]
   local length = bag[class][item][2]
   local sample = sample or torch.Tensor(bag[class][item][2] ,2)
   sample:resize(bag[class][item][2], 2)
   sample:select(2, 1):copy(bag_index:narrow(1, start, length))
   sample:select(2, 2):copy(bag_value:narrow(1, start, length))

   local label = label or torch.Tensor(1)
   label[1] = class

   return sample, label
end

-- Iterator
function Data:iterator(sample, label)
   local bag, bag_index, bag_value =
      self.data.bag, self.data.bag_index, self.data.bag_value
   local sample = sample or torch.Tensor(1, 2)
   local label = label or torch.Tensor(1)

   local class = 1
   local item = 0
   local count = 0

   return function()
      item = item + 1
      if item > bag[class]:size(1) then
         class = class + 1
         item = 1
         if bag[class] == nil then return end
      end
      while bag[class][item][2] == 0 do
         item = item + 1
         if item > bag[class]:size(1) then
            class = class + 1
            item = 1
            if bag[class] == nil then return end
         end
      end
      local start = bag[class][item][1]
      local length = bag[class][item][2]
      sample:resize(length, 2)
      sample:select(2, 1):copy(bag_index:narrow(1, start, length))
      sample:select(2, 2):copy(bag_value:narrow(1, start, length))
      label[1] = class
      return sample, label
   end
end

-- Get data table for share
function Data:getTable()
   return self.data
end

return Data

--[[
Data program for GlyphNet
Copyright 2015-2016 Xiang Zhang
--]]

local class = require('pl.class')
local math = require('math')
local torch = require('torch')

local Data = class()

-- Constructor for Data
-- config: configuration table
--   .file: the data file location
--   .unifont: the unifont data location
--   .length: the text length in the data
--   .batch: the batch size
function Data:_init(config)
   self.data = torch.load(config.file)
   self.unifont = torch.load(config.unifont or 'unifont/unifont-8.0.01.t7b')
   self.length = config.length or 512
   self.batch = config.batch or 16
end

function Data:getClasses()
   return #self.data.code
end

function Data:getBatch(sample, label)
   local code, code_value = self.data.code, self.data.code_value
   local sample, label = self:initSample(sample, label)

   -- Loop over batch dimension
   for i = 1, sample:size(1) do
      local class = torch.random(#code)
      local item = torch.random(code[class]:size(1))

      -- Assign sample
      self:index(sample[i], class, item)
      -- Assign label
      label[i] = class
   end

   return sample, label
end

function Data:iterator(sample, label)
   local code, code_value = self.data.code, self.data.code_value
   local sample, label = self:initSample(sample, label)

   local class = 1
   local item = 1
   local count = 0

   return function()
      if code[class] == nil then return end

      sample, label = self:initSample(sample, label)
      count = 0
      for i = 1, sample:size(1) do
         if item > code[class]:size(1) then
            class = class + 1
            item = 1
            if code[class] == nil then
               if count > 0 then
                  break
               else
                  return
               end
            end
         end
         self:index(sample[i], class, item)
         label[i] = class
         count = count + 1
         item = item + 1
      end

      return sample, label, count
   end
end

function Data:initSample(sample, label)
   local height, width = self.unifont:size(3), self.unifont:size(2)
   local sample = sample or
      torch.Tensor(self.batch, self.length, height, width)
   local label = label or torch.Tensor(self.batch)
   sample:zero()
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
      sample:narrow(1, position, length):index(
         self.unifont, 1, code_value:narrow(
            1, code[class][item][field][1], length))
      position = position + length
   end

   return sample
end

return Data

--[[
Unit test for EmbedNet data component
Copyright 2016 Xiang Zhang
--]]

local Data = require('data')

-- A Logic Named Joe
local joe = {}

function joe.main()
   if joe.init then
      print('Initializing testing environment')
      joe:init()
   end
   for name, func in pairs(joe) do
      if type(name) == 'string' and type(func) == 'function'
      and name:match('[%g]+Test') then
         print('\nExecuting '..name)
         func(joe)
      end
   end
end

function joe:init()
   local config = dofile('config.lua')
   config.train_data.length = 512
   config.test_data.length = 512

   print('Creating testing data object')
   local data = Data(config.test_data)

   self.config = config
   self.data = data
end

function joe:printSample(sample, label, count)
   local count = count or sample:size(1)
   for i = 1, count do
      io.write(label[i], ':')
      for j = 1, sample:size(2) do
         io.write(' ', sample[i][j])
      end
      io.write('\n')
   end
   io.flush()
end

function joe:getBatchTest()
   local data = self.data
   print('Getting a batch')
   local sample, label = data:getBatch()
   self:printSample(sample, label)
   print('Getting a second batch')
   sample, label = data:getBatch(sample, label)
   self:printSample(sample, label)
end

function joe:iteratorTest()
   local data = self.data
   for sample, label, count in data:iterator() do
      io.write(count, ':')
      for i = 1, count do
         io.write(' ', label[i])
      end
      io.write('\n')
      io.flush()
   end
end

joe.main()
return joe

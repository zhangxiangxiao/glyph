--[[
Unit test for LinearNet data program
Copyright 2016 Xiang Zhang
--]]

local Data = require('data')

local math = require('math')
local string = require('string')

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
   config.train_data.file = 'data/dianping/unittest_charbag.t7b'
   self.config = config
   print('Loading data from '..config.train_data.file)
   self.data = Data(config.train_data)
end

function joe:getSampleTest()
   local data = self.data
   print('Getting 10 samples')
   for i = 1, 10 do
      local sample, label = data:getSample(sample, label)
      io.write(label[1], ' ', sample:size(1))
      for j = 1, sample:size(1) do
         io.write(' ', sample[j][1], ':', string.format('%.2g', sample[j][2]))
      end
      io.write('\n')
      io.flush()
   end
end

function joe:iteratorTest()
   local data = self.data
   print('Iterating through data')
   local count = 0
   for sample, label in data:iterator() do
      io.write(label[1], ' ', sample:size(1))
      count = count + 1
      if math.fmod(count, 16) == 0 then
         io.write('\n')
         io.flush()
      else
         io.write(', ')
      end
   end
   if math.fmod(count, 16) ~= 0 then
      io.write('\n')
      io.flush()
   end
end

joe.main()
return joe

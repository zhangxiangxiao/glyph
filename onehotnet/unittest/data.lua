--[[
Unit test for OnehotNet data component
Copyright 2016 Xiang Zhang
--]]

local Data = require('data')

local image = require('image')

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
   config.train_data.length = 2048
   config.test_data.length = 2048

   print('Creating testing data object')
   local data = Data(config.test_data)

   self.config = config
   self.data = data
end

function joe:getBatchTest()
   local data = self.data
   print('Getting a batch')
   local sample, label = data:getBatch()
   local win = image.display{image = sample[1]:narrow(2, 1, 512), zoom = 3}
   print('Getting a second batch')
   sample, label = data:getBatch(sample, label)
   win = image.display{
      win = win, image = sample[1]:narrow(2, 1, 512), zoom = 3}
end

function joe:iteratorTest()
   local data = self.data
   local win
   for sample, label, count in data:iterator() do
      win = image.display{
         win = win, image = sample[1]:narrow(2, 1, 512), zoom = 3}
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

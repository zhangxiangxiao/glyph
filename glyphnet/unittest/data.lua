--[[
Unit test for GlyphNet data program
Copyright 2015-2016 Xiang Zhang
--]]

local Data = require('data')

local image = require('image')

-- A Logic Named Joe
local joe = {}

function joe.main()
   if joe.init then
      print('Initializing testing environment')
      joe.init()
   end
   for name, func in pairs(joe) do
      if type(name) == 'string' and type(func) == 'function'
      and name:match('[%g]+Test') then
         print('\nExecuting '..name)
         func(joe)
      end
   end
end

function joe.init()
   local config = {}
   config.file = 'data/dianping/test_code.t7b'
   config.unifont = 'unifont/unifont-8.0.01.t7b'
   config.length = 512
   config.batch = 16

   joe.config = config
   joe.data = Data(config)
end

function joe.getBatchTest()
   local data = joe.data
   local sample, label = data:getBatch()

   print('Size of sample: ')
   print(sample:size())
   print('Size of label: ')
   print(label:size())

   io.write('Labels:')
   for i = 1, label:size(1) do
      io.write(' ', label[i])
   end
   io.write('\n')

   image.display{image = sample[1]:narrow(1, 1, 100),
                 nrow = 10, zoom = 4}

   joe.sample = sample
   joe.label = label
end

function joe.iteratorTest()
   local data = joe.data

   local window
   local total = 0
   for sample, label, count in data:iterator() do
      total = total + count
      io.write(total, ',', count, ':')
      for i = 1, count do
         window = image.display{
            image = sample[1][1], nrow = 10, zoom = 4, win = window}
         io.write(' ', label[i])
      end
      io.write('\n')
      io.flush()
   end
end

joe.main()
return joe

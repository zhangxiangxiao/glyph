--[[
Unit test for GlyphNet train component
Copyright 2015-2016 Xiang Zhang
--]]

local Train = require('train')

local nn = require('nn')
local os = require('os')

local Data = require('data')
local Model = require('model')

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
   config.test_data.batch = 2
   print('Creating data')
   local data = Data(config.test_data)
   print('Create model')
   local model = Model(config.model)
   print('Create loss')
   local loss = nn[config.driver.loss:sub(4)]()
   print('Create trainer')
   config.train.rates[4] = 1e-5
   local train = Train(data, model, loss, config.train)

   self.data = data
   self.model = model
   self.loss = loss
   self.train = train
   self.config = config
end

function joe:trainTest()
   local train = self.train
   local callback = self:callback()

   print('Running for 10 steps')
   train:run(10, callback)
end

function joe:callback()
   return function (train, i)
      print('stp: '..train.step..', rat: '..train.rate..
               ', obj: '..train.objective..', dat: '..train.time.data..
               ', fwd: '..train.time.forward..', bwd: '..train.time.backward..
               ', upd: '..train.time.update)
   end
end

joe.main()
return joe

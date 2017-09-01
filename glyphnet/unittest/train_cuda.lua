--[[
Unit test for GlyphNet train component
Copyright 2015-2016 Xiang Zhang
--]]

local Train = require('train')

local cutorch = require('cutorch')
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
   print('Setting device to '..config.driver.device)
   cutorch.setDevice(config.driver.device)
   print('Creating data')
   local data = Data(config.test_data)
   print('Create model')
   local model = Model(config.model)
   model:cuda()
   print('Create loss')
   local loss = nn[config.driver.loss:sub(4)]()
   loss:cuda()
   print('Create trainer')
   config.train.rates[79] = 1e-5
   config.train.rates[85] = config.train.rates[1]
   local train = Train(data, model, loss, config.train)

   print('pmn: '..train.params:mean()..', psd: '..train.params:std()..
            ', gmn: '..train.grads:mean()..', gsd: '..train.grads:std()..
            ', smn: '..train.state:mean()..', ssd: '..train.state:std())

   self.data = data
   self.model = model
   self.loss = loss
   self.train = train
   self.config = config
end

function joe:trainTest()
   local train = self.train
   local callback = self:callback()

   print('Running for 100 steps')
   train:run(100, callback)
end

function joe:callback()
   return function (train, i)
      print('stp: '..train.step..', rat: '..train.rate..', err: '..train.error..
               ', obj: '..train.objective..', dat: '..train.time.data..
               ', fwd: '..train.time.forward..', bwd: '..train.time.backward..
               ', upd: '..train.time.update..', pmn: '..train.params:mean()..
               ', psd: '..train.params:std()..', gmn: '..train.grads:mean()..
               ', gsd: '..train.grads:std()..', smn: '..train.state:mean()..
               ', ssd: '..train.state:std())
   end
end

joe.main()
return joe

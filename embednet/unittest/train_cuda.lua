--[[
Unit test for EmbedNet train component
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
   config.test_data.length = config.variation['large'].length
   local data = Data(config.test_data)
   print('Create model')
   config.model.embedding = config.variation['large'].embedding
   config.model.temporal = config.variation['large'].temporal
   local model = Model(config.model)
   model:cuda()
   print('Create loss')
   local loss = nn[config.driver.loss:sub(4)]()
   loss:cuda()
   print('Create trainer')
   for i, v in pairs(config.train.rates) do
      config.train.rates[i] = v * config.driver.rate
   end
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

   print('Running for 100000 steps')
   train:run(100000, callback)
end

function joe:callback()
   self.time = os.time()
   return function (train, i)
      if os.difftime(os.time(), self.time) >= 5 then
         print('stp: '..train.step..', rat: '..train.rate..
                  ', err: '..train.error..', obj: '..train.objective..
                  ', dat: '..train.time.data..', fwd: '..train.time.forward..
                  ', bwd: '..train.time.backward..', upd: '..train.time.update)
         self.time = os.time()
      end
   end
end

joe.main()
return joe

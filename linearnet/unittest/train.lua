--[[
Unit test for LinearNet trainer
Copyright 2016 Xiang Zhang
--]]

local Train = require('train')

local math = require('math')

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
   config.train_data.file = 'data/dianping/unittest_charbag.t7b'
   print('Loading data from '..config.train_data.file)
   self.data = Data(config.train_data)
   print('Loading the model')
   self.model = Model(config.model)
   print(self.model.linear)
   print('Resetting model')
   self.model:reset(1e-2)
   print('Loading the loss')
   self.loss = nn[config.driver.loss:sub(4)]()
   print(self.loss)
   print('Loading the trainer')
   self.train = Train(self.data, self.model, self.loss)
end

function joe:runTest()
   local callback = function(train, step)
      local model = train.model
      if math.fmod(step, 1000) == 0 then
         local max, decision = train.output:max(1)
         print('stp = '..step..
                  ', lbl = '..train.label[1]..
                  ', dcs = '..decision[1]..
                  ', obj = '..train.objective..
                  ', wmn = '..model.linear.weight:mean()..
                  ', wsd = '..model.linear.weight:std()..
                  ', bmn = '..model.linear.bias:mean()..
                  ', bsd = '..model.linear.bias:std())
      end
   end
   local steps = 1000000
   local train = self.train
   print('Training for '..steps..' steps')
   train:run(steps, callback)
end

joe.main()
return joe


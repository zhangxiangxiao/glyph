--[[
Unit test for LinearNet tester
Copyright 2016 Xiang Zhang
--]]

local Test = require('test')

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
   print('Loading the tester')
   self.test = Test(self.data, self.model, self.loss)
end

function joe:runTest()
   local callback = function(test, step)
      print('stp = '..step..
               ', lss = '..test.total_objective..
               ', err = '..test.total_error..
               ', obj = '..test.objective..
               ', lbl = '..test.label[1]..
               ', dcs = '..test.decision[1])
   end
   print('Starting test')
   self.test:run(callback)
end

joe.main()
return joe

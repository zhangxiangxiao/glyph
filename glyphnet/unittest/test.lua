--[[
Unit test for GlyphNet test component
Copyright 2015-2016 Xiang Zhang
--]]

local Test = require('test')

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
   print('Create tester')
   local test = Test(data, model, loss, config.train)

   self.data = data
   self.model = model
   self.loss = loss
   self.test = test
   self.config = config
end

function joe:testTest()
   local test = self.test
   local callback = self:callback()

   print('Running tests')
   test:run(callback)
end

function joe:callback()
   return function (test, i)
      print('cnt: '..test.total_count..', err: '..test.total_error..
               ', lss: '..test.total_objective..', obj: '..test.objective..
               ', crr: '..test.error..', dat: '..test.time.data..
               ', fwd: '..test.time.forward..', upd: '..test.time.update)
   end
end

joe.main()
return joe

--[[
Unit test for driver
Copyright 2016 Xiang Zhang
--]]

local Driver = require('driver')

--  A Logic Named Joe
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

   print('Creating driver')
   config.train_data.file = 'data/dianping/unittest_charbag.t7b'
   config.test_data.file = 'data/dianping/unittest_charbag.t7b'
   config.driver.steps = 10000
   config.driver.epoches = 30
   config.driver.interval = 1
   config.driver.location = '/tmp'
   config.driver.debug = true
   local driver = Driver(config, config.driver)

   self.config = config
   self.driver = driver
end

function joe:driverTest()
   local driver = self.driver
   print('Testing driver')
   driver:run()
end

joe.main()
return joe

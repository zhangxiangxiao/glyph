--[[
Unit Test for OnehotNet model
Copyright 2016 Xiang Zhang
--]]

local Model = require('model')

local sys = require('sys')

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
   config.model.onehot = config.variation['large'].onehot
   config.model.temporal = config.variation['large'].temporal

   local model = Model(config.model)
   print('Onehot model:')
   print(model.onehot)
   print('Temporal model:')
   print(model.temporal)

   self.config = config
   self.model = model
end

function joe:modelTest()
   local model = self.model

   local params, grads = model:getParameters()
   grads:zero()
   print('Number of elements in parameters and gradients: '..
            params:nElement()..', '..grads:nElement())

   print('Creating input')
   local input = torch.rand(2, 256, 2048)
   print(input:size())

   print('Forward propagating')
   sys.tic()
   local output = model:forward(input)
   sys.toc(true)
   print(output:size())

   print('Creating output gradients')
   local grad_output = torch.rand(output:size())
   print(grad_output:size())

   print('Backward propagating')
   sys.tic()
   local grad_input = model:backward(input, grad_output)
   sys.toc(true)
   print(grad_input:size())
end

function joe:modeTest()
   local model = self.model

   print('Setting model to train')
   model:setModeTrain()
   for i, m in ipairs(model.temporal.modules) do
      if torch.type(m) == 'nn.Dropout' then
         print(i, torch.type(m), m.train)
      end
   end

   print('Setting model to test')
   model:setModeTest()
   for i, m in ipairs(model.temporal.modules) do
      if torch.type(m) == 'nn.Dropout' then
         print(i, torch.type(m), m.train)
      end
   end

   print('Setting model to train')
   model:setModeTrain()
   for i, m in ipairs(model.temporal.modules) do
      if torch.type(m) == 'nn.Dropout' then
         print(i, torch.type(m), m.train)
      end
   end

   print('Setting model to test')
   model:setModeTest()
   for i, m in ipairs(model.temporal.modules) do
      if torch.type(m) == 'nn.Dropout' then
         print(i, torch.type(m), m.train)
      end
   end
end

function joe:saveTest()
   local model = self.model
   print('Saving to /tmp/model.t7b')
   model:save('/tmp/model.t7b')

   print('Loading from /tmp/model.t7b')
   local config = self.config
   config.model.file = '/tmp/model.t7b'
   local loaded = Model(config.model)

   print('Onehot model')
   print(loaded.onehot)
   print('Temporal model')
   print(loaded.temporal)

   config.model.file = nil
end

joe.main()
return joe

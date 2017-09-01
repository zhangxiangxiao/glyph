--[[
Unit test for LinearNet model program
Copyright 2016 Xiang Zhang
--]]

local Model = require('model')

local math = require('math')
local string = require('string')
local sys = require('sys')

local Data = require('data')

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
   self.model:reset(1e-3)
   print(self.model.linear.weight:std())
end

function joe:propagationTest()
   local data = self.data
   local model = self.model
   local weight = self.model.linear.weight
   local bias = self.model.linear.bias

   print('Testing forward and backward propagation for 10 times')
   for i = 1, 10 do
      print('Zero gradient of parameters')
      sys.tic()
      model:zeroGradParameters()
      sys.toc(true)

      local sample, label = data:getSample()
      print(tostring(i)..', sample '..sample:size(1)..', label '..label[1])
      print('Forward propagating')
      sys.tic()
      local output = model:forward(sample)
      sys.toc(true)

      print('output '..output:dim()..', '..output:size(1))
      print('Backward propagating')
      local grad_output = torch.rand(output:size())
      sys.tic()
      local grad_input = model:backward(sample, grad_output)
      sys.toc(true)
      print('grad_input '..tostring(grad_input))

      print('Update parameters')
      sys.tic()
      model:updateParameters(1e-3)
      sys.toc(true)
      print('weight mean '..weight:mean()..', std '..weight:std()..
               ', bias mean '..bias:mean()..', std '..bias:std())
   end
end

function joe:shareModuleTest()
   local model = self.model
   local linear = model.linear:clone()
   print(torch.pointer(model.linear.weight:storage()),
         torch.pointer(linear.weight:storage()),
         torch.pointer(model.linear.bias:storage()),
         torch.pointer(linear.bias:storage()))
   model:shareModules({linear = linear})
   print(torch.pointer(model.linear.weight:storage()),
         torch.pointer(linear.weight:storage()),
         torch.pointer(model.linear.bias:storage()),
         torch.pointer(linear.bias:storage()))
end

function joe:saveTest()
   local model = self.model
   local weight, bias = model.linear.weight, model.linear.bias
   print('weight mean '..weight:mean()..', std '..weight:std()..
            ', bias mean '..bias:mean()..', std '..bias:std())
   print('Saving model to /tmp/model.t7b')
   model:save('/tmp/model.t7b')
   print('Resetting model with sigma 1e-2')
   model:reset(1e-2)
   print('weight mean '..weight:mean()..', std '..weight:std()..
            ', bias mean '..bias:mean()..', std '..bias:std())
   print('Loading model from /tmp/model.t7b')
   model:load('/tmp/model.t7b')
   print('weight mean '..weight:mean()..', std '..weight:std()..
            ', bias mean '..bias:mean()..', std '..bias:std())
end

joe.main()
return joe

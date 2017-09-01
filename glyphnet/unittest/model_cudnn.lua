--[[
Unit test for GlyphNet model component
Copyright 2015-2016 Xiang Zhang
--]]

local Model = require('model')

local cutorch = require('cutorch')
local os = require('os')
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
   config.model.cudnn = true

   print('Changing device to '..config.driver.device)
   cutorch.setDevice(config.driver.device)

   local model = Model(config.model)
   model:cuda()

   local parameters, gradients = model:getParameters()
   print('Parameter pointers: '..torch.pointer(parameters:storage())..' '..
            torch.pointer(gradients:storage()))
   print('Parameter sizes: '..parameters:nElement()..' '..gradients:nElement())

   self.config = config
   self.model = model
   self.parameters = parameters
   self.gradients = gradients

   self:printModel()
end

function joe:printModel(model)
   local model = model or self.model
   print('Type of model: '..model:type())
   print('Created spatial model: ')
   print(model.spatial)
   print('Created temporal model: ')
   print(model.temporal)
   print('Spatial group pointers:')
   print(0, torch.pointer(model.spatial.modules[1].weight:storage()),
         torch.pointer(model.spatial.modules[1].gradWeight:storage()))
   for i, m in ipairs(model.group) do
      print(i, torch.pointer(m.modules[1].weight:storage()),
            torch.pointer(m.modules[1].gradWeight:storage()))
   end
end

function joe:forwardBackwardTest()
   local model = self.model

   print('Initializing input')
   local input = torch.rand(16, 512, 16, 16):type(model:type())
   print('Input size:')
   print(input:size())

   print('Running forward propagation')
   cutorch.synchronize()
   sys.tic()
   local output = model:forward(input)
   cutorch.synchronize()
   sys.toc(true)
   print('Feature size:')
   print(model.feature:size())
   print('Output size:')
   print(output:size())

   print('Initializing output gradients')
   local grad_output = torch.rand(output:size()):type(model:type())
   print('Running backward propagation')
   cutorch.synchronize()
   sys.tic()
   local grad_input = model:backward(input, grad_output)
   cutorch.synchronize()
   sys.toc(true)
   print('Feature gradient size:')
   print(model.grad_feature:size())

   self.input = input
   self.grad_input = grad_input
   self.output = output
   self.grad_output = grad_output
end

function joe:saveTest()
   local model = self.model
   local file = '/tmp/model.t7b'
   print('Saving to '..file)
   model:save(file)
   print('Model saved')

   local config = {}
   config.file = file
   config.cudnn = joe.config.model.cudnn
   config.group = joe.config.model.group
   print('Loading from '..file)
   model = Model(config)

   self:printModel(model)
end

function joe:modeTest()
   local model = self.model
   print('Setting to testing mode')
   model:setModeTest()
   print('Temporal mode:')
   for i, m in ipairs(model.temporal.modules) do
      print(i, torch.type(m), m.train)
   end
   print('Spatial mode:')
   for i, m in ipairs(model.spatial.modules) do
      print(i, torch.type(m), m.train)
   end

   print('Setting to training mode')
   model:setModeTrain()
   print('Temporal mode:')
   for i, m in ipairs(model.temporal.modules) do
      print(i, torch.type(m), m.train)
   end
   print('Spatial mode:')
   for i, m in ipairs(model.spatial.modules) do
      print(i, torch.type(m), m.train)
   end
end

joe.main()
return joe

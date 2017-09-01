--[[
Training class for LinearNet
Copyright 2016 Xiang Zhang
--]]

local class = require('pl.class')
local math = require('math')
local nn = require('nn')

local Train = class()

-- Constructor
-- data: the data instance
-- model: the model instance
-- loss: the loss instance
-- config: the configuration table
--   .rate: learning rate
--   .step: current finished steps. Starting from 0
function Train:_init(data, model, loss, config)
   self.data = data
   self.model = model
   self.loss = loss

   local config = config or {}
   self.rate = config.rate or 1e-3
   self.step = config.step or 0

   self.type = model:type()
end

-- Run for a number of steps
-- steps: number of steps to run
-- callback: a function to execute after each step
function Train:run(steps, callback)
   for i = 1, steps do
      self:runStep()
      self.step = self.step + 1
      if callback then
         callback(self, i)
      end
   end
end

-- Run for one step
function Train:runStep()
   -- Get sample
   self.sample, self.label = self.data:getSample(self.sample, self.label)

   -- Forward propagation
   self.output = self.model:forward(self.sample)
   self.objective = self.loss:forward(self.output, self.label)

   -- Backward propagation
   self.grad_output = self.loss:backward(self.output, self.label)
   self.grad_input = self.model:backward(self.sample, self.grad_output)

   -- Update parameters
   self.model:updateParameters(self.rate)
   self.model:zeroGradParameters()
end

return Train

--[[
Tester for LinearNet
Copyright 2016 Xiang Zhang
--]]

local class = require('pl.class')
local math = require('math')
local torch = require('torch')

local Test = class()

-- Constructor
-- data: the data instance
-- model: the model instance
-- loss: the loss instance
-- config: configuration table
function Test:_init(data, model, loss, config)
   self.data = data
   self.model = model
   self.loss = loss

   self.type = model:type()
end

-- Run the tester
-- callback: a function to execute after each step
function Test:run(callback)
   self.total_objective = 0
   self.total_error = 1
   self.step = 0
   for sample, label in self.data:iterator() do
      self:runStep(sample, label)
      self.step = self.step + 1
      if callback then
         callback(self, self.step)
      end
   end
end

-- Run for one step
function Test:runStep(sample, label)
   -- Get sample
   self.sample, self.label = sample, label

   -- Forward propagation
   self.output = self.model:forward(self.sample)
   self.objective = self.loss:forward(self.output, self.label)

   -- Compute decision
   self.max, self.decision = self.output:max(1)
   self.error = (self.decision[1] == self.label[1]) and 0 or 1

   -- Accumulate errors
   self.total_objective = (self.total_objective * self.step + self.objective) /
      (self.step + 1)
   self.total_error = (self.total_error * self.step + self.error) /
      (self.step + 1)
end

return Test

--[[
Tester for GlyphNet
Copyright 2016 Xiang Zhang
--]]

local class = require('pl.class')
local math = require('math')
local torch = require('torch')
local sys = require('sys')

local Test = class()

-- Constructor for Test
-- data: the data object
-- model: the model object
-- loss: the loss object
-- config: configuration table
function Test:_init(data, model, loss, config)
   self.data = data
   self.model = model
   self.loss = loss

   self.time = {}
end

-- Run for all the data
-- callback: (optional) a callback function to execute after each step
function Test:run(callback)
   self.total_error = 0
   self.total_objective = 0
   self.total_count = 0
   self.clock = sys.clock()
   for input, label, count in self.data:iterator() do
      self:runStep(input, label, count)
      if callback then callback(self) end
      self.clock = sys.clock()
   end
end

-- Run for one minibatch step
function Test:runStep(input, label, count)
   -- Get a batch of data
   self.input_untyped, self.label_untyped = input, label
   self.input = self.input or self.input_untyped:type(self.model:type())
   self.input:copy(self.input_untyped)
   self.label = self.label or self.label_untyped:type(self.model:type())
   self.label:copy(self.label_untyped)
   self.count = count
   if self.model:type() == 'torch.CudaTensor' then cutorch.synchronize() end
   self.time.data = sys.clock() - self.clock

   -- Forward propagation
   self.clock = sys.clock()
   self.output = self.model:forward(self.input)
   self.objective = self.loss:forward(self.output, self.label)
   if type(self.objective) ~= 'number' then self.objective = self.objectve[1] end
   self.max, self.decision = self.output:type(
      torch.getdefaulttensortype()):max(2)
   self.max = self.max:squeeze()
   self.decision = self.decision:squeeze():narrow(1, 1, count):type(
      torch.getdefaulttensortype())
   self.error = torch.ne(
      self.decision, self.label_untyped:narrow(1, 1, count)):type(
      torch.getdefaulttensortype()):sum() / count
   if self.model:type() == 'torch.CudaTensor' then cutorch.synchronize() end
   self.time.forward = sys.clock() - self.clock

   -- Update the results
   self.clock = sys.clock()
   self.total_objective =
      (self.total_objective * self.total_count + self.objective * count) /
      (self.total_count + count)
   self.total_error =
      (self.total_error * self.total_count + self.error * count) /
      (self.total_count + count)
   self.total_count = self.total_count + count
   self.time.update = sys.clock() - self.clock
end

return Test

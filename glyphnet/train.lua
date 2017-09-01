--[[
Trainer for GlyphNet
Copyright 2016 Xiang Zhang
--]]

local class = require('pl.class')
local math = require('math')
local torch = require('torch')
local sys = require('sys')

local Train = class()

-- Constructor for Train
-- data: the data object
-- model: the model object
-- loss: the loss object
-- config: configuration table
function Train:_init(data, model, loss, config)
   self.data = data
   self.model = model
   self.loss = loss

   self.rates = config.rates or {1e-3}
   self.step = config.step or 0
   self.momentum = config.momentum or 0
   self.decay = config.decay or 0
   self.recapture = config.recapture

   self.params, self.grads = self.model:getParameters()
   if config.state then
      self.state = config.state:type(self.model:type())
   else
      self.state = self.grads:clone():zero()
   end

   -- Find current learning rate
   local max_step = 1
   self.rate = self.rates[1]
   for step, rate in pairs(self.rates) do
      if step <= self.step and step > max_step then
         max_step = step
         self.rate = rate
      end
   end

   self.time = {}
end

-- Run for a number of steps
-- steps: number of steps
-- callback: (optional) a callback function to execute after each step
function Train:run(steps, callback)
   if self.recapture then
      self.params, self.grads = self.model:getParameters()
   end

   for i = 1, steps do
      self.step = self.step + 1
      self:runStep()
      if callback then callback(self, i) end
   end
end

-- Run for one minibatch step
function Train:runStep()
   -- Get a batch of data/
   self.clock = sys.clock()
   self.input_untyped, self.label_untyped = self.data:getBatch(
      self.input_untyped, self.label_untyped)
   self.input = self.input or self.input_untyped:type(self.model:type())
   self.input:copy(self.input_untyped)
   self.label = self.label or self.label_untyped:type(self.model:type())
   self.label:copy(self.label_untyped)
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
   self.decision = self.decision:squeeze():type(torch.getdefaulttensortype())
   self.error = torch.ne(self.decision, self.label_untyped):type(
      torch.getdefaulttensortype()):sum() / self.label:size(1)
   if self.model:type() == 'torch.CudaTensor' then cutorch.synchronize() end
   self.time.forward = sys.clock() - self.clock

   -- Backward propagation
   self.clock = sys.clock()
   self.grads:zero()
   self.grad_output = self.loss:backward(self.output, self.label)
   self.grad_input = self.model:backward(self.input, self.grad_output)
   if self.model:type() == 'torch.CudaTensor' then cutorch.synchronize() end
   self.time.backward = sys.clock() - self.clock

   -- Update the step
   self.clock = sys.clock()
   self:sgd()
   if self.model:type() == 'torch.CudaTensor' then cutorch.synchronize() end
   self.time.update = sys.clock() - self.clock
end

function Train:sgd()
   self.rate = self.rates[self.step] or self.rate
   if self.momentum and self.momentum > 0 then
      self.state:mul(self.momentum):add(self.grads:mul(-self.rate))
      self.params:mul(1 - self.rate * self.decay):add(self.state)
   else
      self.params:mul(1 - self.rate * self.decay):add(
         self.grads:mul(-self.rate))
   end
end

return Train

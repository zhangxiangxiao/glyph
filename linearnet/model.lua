--[[
Model class for LinearNet, using SparseLinear
Copyright 2016 Xiang Zhang
--]]

local class = require('pl.class')
local nn = require('nn')
local torch = require('torch')

local Model = class()

-- Constructor for model
-- config: configuration table
--   .size: size of input index
--   .dimension: dimension of output
--   .decay: weight decay. Optional.
-- modules: share weights with the given modules. Optional.
function Model:_init(config, modules)
   self.size = config.size
   self.dimension = config.dimension
   self.decay = config.decay or 0

   if modules then
      self.linear = modules.linear:clone('weight', 'bias')
   else
      self.linear = nn.SparseLinear(self.size, self.dimension)
   end

   self.sequential = nn.Sequential()
   self.sequential:add(self.linear)
   self.sequential:add(nn.LogSoftMax())
end

-- Forward propagation
function Model:forward(input)
   return self.sequential:forward(input)
end

-- Backward propagation
function Model:backward(input, grad_output)
   local grad_input = self.sequential:backward(input, grad_output)
   -- Apply weight decay to linear module
   if self.decay > 0 then
      self.linear_index = self.linear_index or torch.LongTensor(input:size(1))
      self.linear_index:resize(input:size(1)):copy(input:select(2, 1))
      self.linear_decay = self.linear_decay or self.linear.gradWeight:new()
      self.linear_decay:index(self.linear.weight, 2, self.linear_index)
      self.linear.gradWeight:indexAdd(
         2, self.linear_index, self.linear_decay:mul(self.decay))
      self.linear.gradBias:add(self.decay, self.linear.bias)
   end
   return grad_input
end

-- Update parameters
function Model:updateParameters(rate)
   return self.linear:updateParameters(rate)
end

-- Zero grad parameters
function Model:zeroGradParameters()
   return self.linear:zeroGradParameters()
end

-- Set the type
function Model:type(tensortype)
   local tensortype = tensortype or self.linear.weight:type()
   if tensor_type ~= self.linear.weight:type() then
      self.linear:type(tensortype)
   end
   return tensortype
end

-- Reset the weights
function Model:reset(sigma)
   self.linear.weight:normal(0, sigma)
   self.linear.bias:zero()
end

-- Get the modules
function Model:getModules()
   return {linear = self.linear}
end

-- Share given modules
function Model:shareModules(modules)
   self.linear:share(modules.linear, 'weight', 'bias')
end

-- Save to file
function Model:save(file)
   torch.save(file, self.linear)
end

-- Load from file
function Model:load(file)
   local linear = torch.load(file)
   self.linear.weight:copy(linear.weight)
   self.linear.bias:copy(linear.bias)
end

return Model

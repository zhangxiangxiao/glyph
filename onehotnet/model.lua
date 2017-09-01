--[[
Model for OnehotNet
Copyright 2016 Xiang Zhang
--]]

local class = require('pl.class')
local nn = require('nn')

local parent = require('glyphnet/model')

local Model = class(parent)

-- Model constructor
-- config: configuration table
--   .onehot: configuration table of the onehot model
--   .temporal: configuration table of the temporal model
--   .file: the model file to load
--   .cudnn: whether to use NVidia CUDNN
function Model:_init(config)
   -- Read or create model
   if config.file then
      local model = torch.load(config.file)
      self.onehot = self:makeCleanSequential(model.onehot)
      self.temporal = self:makeCleanSequential(model.temporal)
   else
      self.onehot = self:createCleanSequential(config.onehot)
      self:initSequential(self.onehot)
      self.temporal = self:createCleanSequential(config.temporal)
      self:initSequential(self.temporal)
   end

   -- Saving configurations
   self.cudnn = config.cudnn
   self.config = config
   self.tensortype = torch.getdefaulttensortype()
end

function Model:forward(input)
   self.feature = self.onehot:forward(input)
   self.output = self.temporal:forward(self.feature)
   return self.output
end

function Model:backward(input, grad_output)
   self.grad_feature = self.temporal:backward(self.feature, grad_output)
   self.grad_input = self.onehot:backward(input, self.grad_feature)
   return self.grad_input
end

function Model:getParameters()
   return nn.Module.getParameters(self)
end

function Model:parameters()
   local parameters, gradients = {}, {}

   if not self.pretrain then
      local onehot_parameters, onehot_gradients =
         self.onehot:parameters()
      for i = 1, #onehot_parameters do
         parameters[#parameters + 1] = onehot_parameters[i]
         gradients[#gradients + 1] = onehot_gradients[i]
      end
   end

   local temporal_parameters, temporal_gradients = self.temporal:parameters()
   for i = 1, #temporal_parameters do
      parameters[#parameters + 1] = temporal_parameters[i]
      gradients[#gradients + 1] = temporal_gradients[i]
   end

   return parameters, gradients
end

function Model:type(tensortype)
   if tensortype ~= nil and tensortype ~= self.tensortype then
      if tensortype == 'torch.CudaTensor' then
         require('cunn')
         self.onehot = self:makeCudaSequential(self.onehot)
         self.temporal = self:makeCudaSequential(self.temporal)
      else
         self.onehot = self:makeCleanSequential(self.onehot)
         self.temporal = self:makeCleanSequential(self.temporal)
      end
      self.onehot:type(tensortype)
      self.temporal:type(tensortype)
      self.tensortype = tensortype
   end

   return self.tensortype
end

function Model:setMode(mode)
   self:setModeSequential(self.onehot, mode)
   self:setModeSequential(self.temporal, mode)
end


function Model:save(file)
   local onehot = self:clearSequential(
      self:makeCleanSequential(self.onehot))
   local temporal = self:clearSequential(
      self:makeCleanSequential(self.temporal))
   torch.save(file, {onehot = onehot, temporal = temporal})
end

return Model

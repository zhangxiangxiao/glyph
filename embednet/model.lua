--[[
Model for EmbedNet
Copyright 2016 Xiang Zhang
--]]

local class = require('pl.class')
local nn = require('nn')

local parent = require('glyphnet/model')

local Model = class(parent)

-- Model constructor
-- config: configuration table
--   .embedding: configuration table of the embedding model
--   .temporal: configuration table of the temporal model
--   .file: the model file to load
--   .pretrain: whether the keep the embedding pretrained
--   .embedding_file: the file for pretrained embedding model
--   .cudnn: whether to use NVidia CUDNN
function Model:_init(config)
   -- Read or create model
   if config.file then
      local model = torch.load(config.file)
      self.embedding = self:makeCleanSequential(model.embedding)
      self.temporal = self:makeCleanSequential(model.temporal)
   else
      if config.embedding_file then
         self.embedding = self:makeCleanSequential(
            torch.load(config.embedding_file))
      else
         self.embedding = self:createCleanSequential(config.embedding)
         self:initSequential(self.embedding)
      end
      self.temporal = self:createCleanSequential(config.temporal)
      self:initSequential(self.temporal)
   end

   -- Saving configurations
   self.pretrain = config.pretrain
   self.cudnn = config.cudnn
   self.config = config
   self.tensortype = torch.getdefaulttensortype()
end

function Model:forward(input)
   self.feature = self.embedding:forward(input)
   self.output = self.temporal:forward(self.feature)
   return self.output
end

function Model:backward(input, grad_output)
   self.grad_feature = self.temporal:backward(self.feature, grad_output)
   if self.pretrain then
      return self.grad_feature
   else
      self.grad_input = self.embedding:backward(input, self.grad_feature)
      return self.grad_input
   end
end

function Model:getParameters()
   return nn.Module.getParameters(self)
end

function Model:parameters()
   local parameters, gradients = {}, {}

   if not self.pretrain then
      local embedding_parameters, embedding_gradients =
         self.embedding:parameters()
      for i = 1, #embedding_parameters do
         parameters[#parameters + 1] = embedding_parameters[i]
         gradients[#gradients + 1] = embedding_gradients[i]
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
         self.embedding = self:makeCudaSequential(self.embedding)
         self.temporal = self:makeCudaSequential(self.temporal)
      else
         self.embedding = self:makeCleanSequential(self.embedding)
         self.temporal = self:makeCleanSequential(self.temporal)
      end
      self.embedding:type(tensortype)
      self.temporal:type(tensortype)
      self.tensortype = tensortype
   end

   return self.tensortype
end

function Model:setMode(mode)
   self:setModeSequential(self.embedding, mode)
   self:setModeSequential(self.temporal, mode)
end


function Model:save(file)
   local embedding = self:clearSequential(
      self:makeCleanSequential(self.embedding))
   local temporal = self:clearSequential(
      self:makeCleanSequential(self.temporal))
   torch.save(file, {embedding = embedding, temporal = temporal})
end

Model.initModule['nn.LookupTable'] = function (self, m)
   m.weight:normal(0, math.sqrt(1 / m.weight:size(2)))
   if m.paddingValue > 0 then
      m.weight[m.paddingValue]:zero()
   end
end
Model.initModule['nn.Transpose'] = function (self, m) end

Model.setModeModule['train']['nn.LookupTable'] = function (self, m) end
Model.setModeModule['train']['nn.Transpose'] = function (self, m) end

Model.setModeModule['test']['nn.LookupTable'] = function(self, m) end
Model.setModeModule['test']['nn.Transpose'] = function(self, m) end

Model.createCleanModule['nn.LookupTable'] = function (self, m)
   return nn.LookupTable(m.nIndex, m.nOutput, m.paddingValue)
end
Model.createCleanModule['nn.Transpose'] = function (self, m)
   return nn.Transpose(unpack(m.permutations))
end

Model.makeCleanModule['nn.LookupTable'] = function(self, m)
   local new = nn.LookupTable(
      m.weight:size(1), m.weight:size(2), m.paddingValue)
   new.weight:copy(m.weight)
   return new
end
Model.makeCleanModule['nn.Transpose'] = function (self, m)
   return nn.Transpose(unpack(m.permutations))
end

Model.makeCudaModule['nn.LookupTable'] = function (self, m)
   local new = nn.LookupTable(
      m.weight:size(1), m.weight:size(2), m.paddingValue)
   new.weight:copy(m.weight)
   return new
end
Model.makeCudaModule['nn.Transpose'] = function (self, m)
   return nn.Transpose(unpack(m.permutations))
end

return Model

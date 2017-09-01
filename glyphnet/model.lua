--[[
Model for GlyphNet
Copyright 2016 Xiang Zhang
--]]

local class = require('pl.class')
local cudnn
local nn = require('nn')
local torch = require('torch')

local Modules = require('modules')

local Model = class()

-- Model constructor
-- config: configuration table
--   .spatial: configuration table of the spatial network
--   .temporal: configuration table of the temporal network
--   .file: (optional) the model file
--   .cudnn: (optional) whether to use NVidia cudnn
--   .group: (optional) number of spatial network groups
function Model:_init(config)
   -- Read or create model
   if config.file then
      local model = torch.load(config.file)
      self.spatial = self:makeCleanSequential(model.spatial)
      self.temporal = self:makeCleanSequential(model.temporal)
   else
      self.spatial = self:createCleanSequential(config.spatial)
      self.temporal = self:createCleanSequential(config.temporal)
      self:initSequential(self.spatial)
      self:initSequential(self.temporal)
   end

   -- Saving configurations
   self.cudnn = config.cudnn
   self.config = config
   self.tensortype = torch.getdefaulttensortype()

   -- Initialize intermediate values
   self.feature = torch.Tensor()
   self.feature_cache = torch.Tensor()
   self.grad_feature = torch.Tensor()
   self.grad_input = torch.Tensor()

   -- Initialize groups
   self:initGroup(config.group)
end

function Model:initGroup(group)
   local group = group or 1

   -- Clean current network group
   if self.group then
      self.group = nil
      collectgarbage()
   end

   -- Create new group
   self.group = {}
   for i = 1, group do
      self.group[i] = self.spatial:clone(
         'weight', 'bias', 'gradWeight', 'gradBias')
   end
end

function Model:forward(input)
   -- Do forward propagation for spatial model group
   local input_group = input:view(
      #self.group, -1, 1, input:size(3), input:size(4))
   local feature = self.group[1]:forward(input_group[1])
   self.feature_cache:resize(#self.group, feature:size(1), feature:size(2))
   self.feature_cache[1]:copy(feature)
   for i = 2, #self.group do
      local feature = self.group[i]:forward(input_group[i])
      self.feature_cache[i]:copy(feature)
   end

   -- Do forward propagation for temporal model
   self.feature:resize(
      input:size(1), self.feature_cache:size(3), input:size(2)):copy(
      self.feature_cache:view(
         input:size(1), input:size(2), self.feature_cache:size(3)):transpose(
         2, 3))
   self.output = self.temporal:forward(self.feature)

   return self.output
end

function Model:backward(input, grad_output)
   -- Do backward propagation for temporal model
   local grad_feature = self.temporal:backward(self.feature, grad_output)
   self.grad_feature:resizeAs(self.feature_cache):view(
      input:size(1), input:size(2), self.feature_cache:size(3)):copy(
      grad_feature:transpose(2, 3)):div(input:size(2))

   -- Do backward propagation for spatial model group
   local input_group = input:view(
      #self.group, -1, 1, input:size(3), input:size(4))
   self.grad_input:resizeAs(input)
   local grad_input_group = self.grad_input:view(
      #self.group, -1, 1, input:size(3), input:size(4))
   for i = 1, #self.group do
      local grad_input = self.group[i]:backward(
         input_group[i], self.grad_feature[i])
      grad_input_group[i]:copy(grad_input)
   end

   return self.grad_input
end

function Model:getParameters()
   local parameters, gradients = nn.Module.getParameters(self)
   self:initGroup(#self.group)
   return parameters, gradients
end

function Model:parameters()
   local parameters, gradients = {}, {}

   local spatial_parameters, spatial_gradients = self.spatial:parameters()
   for i = 1, #spatial_parameters do
      parameters[#parameters + 1] = spatial_parameters[i]
      gradients[#gradients + 1] = spatial_gradients[i]
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
         self.spatial = self:makeCudaSequential(self.spatial)
         self.temporal = self:makeCudaSequential(self.temporal)
      else
         self.spatial = self:makeCleanSequential(self.spatial)
         self.temporal = self:makeCleanSequential(self.temporal)
      end
      self.spatial:type(tensortype)
      self.temporal:type(tensortype)
      self.feature = self.feature:type(tensortype)
      self.feature_cache = self.feature_cache:type(tensortype)
      self.grad_feature = self.grad_feature:type(tensortype)
      self.grad_input = self.grad_input:type(tensortype)
      self.tensortype = tensortype
      self:initGroup(#self.group)
   end

   return self.tensortype
end

function Model:cuda()
   return self:type('torch.CudaTensor')
end

function Model:double()
   return self:type('torch.DoubleTensor')
end

function Model:float()
   return self:type('torch.FloatTensor')
end

function Model:setMode(mode)
   self:setModeSequential(self.temporal, mode)
   self:setModeSequential(self.spatial, mode)
   for i = 1, #self.group do
      self:setModeSequential(self.group[i], mode)
   end
end

function Model:setModeTrain()
   self:setMode('train')
end

function Model:setModeTest()
   self:setMode('test')
end

function Model:save(file)
   local spatial = self:clearSequential(
      self:makeCleanSequential(self.spatial))
   local temporal = self:clearSequential(
      self:makeCleanSequential(self.temporal))
   torch.save(file, {spatial = spatial, temporal = temporal})
end

-- Clear sequential model
function Model:clearSequential(sequential)
   local function recursiveClear(key, param)
      local param = param
      if torch.type(param) == 'table' then
         for k, v in pairs(param) do
            param[k] = recursiveClear(k, v)
         end
      elseif torch.isTensor(param) and key ~= 'weight' and key ~= 'bias' then
         param = param.new()
      end
      return param
   end

   for _, m in ipairs(sequential.modules) do
      for k, v in pairs(m) do
         m[k] = recursiveClear(k, v)
      end
   end

   return sequential
end

-- Initialize sequential using microsoft initialization
function Model:initSequential(sequential)
   for _, m in ipairs(sequential.modules) do
      self.initModule[torch.type(m)](self, m)
   end
end

-- Setting the mode of sequential modules
function Model:setModeSequential(sequential, mode)
   for _, m in ipairs(sequential.modules) do
      self.setModeModule[mode][torch.type(m)](self, m)
   end
end

-- Create a clean sequential
function Model:createCleanSequential(config)
   local new = nn.Sequential()
   for _, m in ipairs(config) do
      new:add(self.createCleanModule[m.name](self, m))
   end
   return new
end

-- Make a clean sequential
function Model:makeCleanSequential(sequential)
   local new = nn.Sequential()
   for _, m in ipairs(sequential.modules) do
      new:add(self.makeCleanModule[torch.type(m)](self, m))
   end
   return new
end

-- Make a CUDA sequential
function Model:makeCudaSequential(sequential)
   if self.cudnn then
      cudnn = require('cudnn')
   end
   local new = nn.Sequential()
   for _, m in ipairs(sequential.modules) do
      new:add(self.makeCudaModule[torch.type(m)](self, m))
   end
   return new
end

-- Initialize modules
Model.initModule = {}
Model.initModule['nn.LogSoftMax'] = function (self, m) end
Model.initModule['nn.Threshold'] = function (self, m) end
Model.initModule['nn.Reshape'] = function (self, m) end
Model.initModule['nn.Dropout'] = function (self, m) end
Model.initModule['nn.Linear'] = function (self, m)
   m.bias:zero()
   m.weight:normal(0, math.sqrt(2 / m.weight:size(1)))
end
Model.initModule['nn.SpatialConvolution'] = function (self, m)
   m.bias:zero()
   m.weight:normal(
      0, math.sqrt(2 / m.weight:size(1) / m.weight:size(3) / m.weight:size(4)))
end
Model.initModule['nn.SpatialMaxPooling'] = function (self, m) end
Model.initModule['nn.TemporalConvolutionMM'] = function (self, m)
   m.bias:zero()
   m.weight:normal(0, math.sqrt(2 / m.weight:size(1) / m.weight:size(3)))
end
Model.initModule['nn.TemporalMaxPoolingMM'] = function (self, m) end

-- Set module mode to train
Model.setModeModule = {}
Model.setModeModule['train'] = {}
Model.setModeModule['train']['nn.LogSoftMax'] = function (self, m) end
Model.setModeModule['train']['cudnn.LogSoftMax'] =
   Model.setModeModule['train']['nn.LogSoftMax']
Model.setModeModule['train']['nn.Threshold'] = function (self, m) end
Model.setModeModule['train']['nn.Reshape'] = function (self, m) end
Model.setModeModule['train']['nn.Dropout'] = function (self, m)
   m.train = true
end
Model.setModeModule['train']['nn.Linear'] = function (self, m) end
Model.setModeModule['train']['nn.SpatialConvolution'] = function (self, m) end
Model.setModeModule['train']['cudnn.SpatialConvolution'] =
   Model.setModeModule['train']['nn.SpatialConvolution']
Model.setModeModule['train']['nn.SpatialMaxPooling'] = function (self, m) end
Model.setModeModule['train']['cudnn.SpatialMaxPooling'] =
   Model.setModeModule['train']['nn.SpatialMaxPooling']
Model.setModeModule['train']['nn.TemporalConvolutionMM'] =
   function (self, m) end
Model.setModeModule['train']['cudnn.TemporalConvolutionCudnn'] =
   function (self, m) end
Model.setModeModule['train']['nn.TemporalMaxPoolingMM'] = function (self, m) end
Model.setModeModule['train']['cudnn.TemporalMaxPoolingCudnn'] =
   Model.setModeModule['train']['nn.TemporalMaxPoolingMM']

-- Set module mode to test
Model.setModeModule['test'] = {}
Model.setModeModule['test']['nn.LogSoftMax'] = function (self, m) end
Model.setModeModule['test']['cudnn.LogSoftMax'] =
   Model.setModeModule['test']['nn.LogSoftMax']
Model.setModeModule['test']['nn.Threshold'] = function (self, m) end
Model.setModeModule['test']['nn.Reshape'] = function (self, m) end
Model.setModeModule['test']['nn.Dropout'] = function (self, m)
   m.train = false
end
Model.setModeModule['test']['nn.Linear'] = function (self, m) end
Model.setModeModule['test']['nn.SpatialConvolution'] = function (self, m) end
Model.setModeModule['test']['cudnn.SpatialConvolution'] =
   Model.setModeModule['test']['nn.SpatialConvolution']
Model.setModeModule['test']['nn.SpatialMaxPooling'] = function (self, m) end
Model.setModeModule['test']['cudnn.SpatialMaxPooling'] =
   Model.setModeModule['test']['nn.SpatialMaxPooling']
Model.setModeModule['test']['nn.TemporalConvolutionMM'] =
   function (self, m) end
Model.setModeModule['test']['cudnn.TemporalConvolutionCudnn'] =
   function (self, m) end
Model.setModeModule['test']['nn.TemporalMaxPoolingMM'] = function (self, m) end
Model.setModeModule['test']['cudnn.TemporalMaxPoolingCudnn'] =
   Model.setModeModule['test']['nn.TemporalMaxPoolingMM']

-- Create clean modules
Model.createCleanModule = {}
Model.createCleanModule['nn.LogSoftMax'] = function (self, m)
   return nn.LogSoftMax()
end
Model.createCleanModule['nn.Threshold'] = function (self, m)
   return nn.Threshold(m.th, m.v, m.ip)
end
Model.createCleanModule['nn.Reshape'] = function (self, m)
   return nn.Reshape(m.size, m.batchMode)
end
Model.createCleanModule['nn.Dropout'] = function (self, m)
   return nn.Dropout(m.p, not m.v2, m.inplace)
end
Model.createCleanModule['nn.Linear'] = function (self, m)
   return nn.Linear(m.inputSize, m.outputSize, m.bias)
end
Model.createCleanModule['nn.SpatialConvolution'] = function (self, m)
   return nn.SpatialConvolution(
      m.nInputPlane, m.nOutputPlane, m.kW, m.kH, m.dW, m.dH, m.padW, m.padH)
end
Model.createCleanModule['nn.SpatialMaxPooling'] = function (self, m)
   return nn.SpatialMaxPooling(m.kW, m.kH, m.dW, m.dH, m.padW, m.padH)
end
Model.createCleanModule['nn.TemporalConvolutionMM'] = function (self, m)
   return nn.TemporalConvolutionMM(
      m.inputFrameSize, m.outputFrameSize, m.kW, m.dW, m.padW)
end
Model.createCleanModule['nn.TemporalMaxPoolingMM'] = function (self, m)
   return nn.TemporalMaxPoolingMM(m.kW, m.dW)
end

-- Make clean modules
Model.makeCleanModule = {}
Model.makeCleanModule['nn.LogSoftMax'] = function (self, m)
   return nn.LogSoftMax()
end
Model.makeCleanModule['cudnn.LogSoftMax'] =
   Model.makeCleanModule['nn.LogSoftMax']
Model.makeCleanModule['nn.Threshold'] = function (self, m)
   return nn.Threshold(m.threshold, m.val, m.inplace)
end
Model.makeCleanModule['nn.Reshape'] = function (self, m)
   return nn.Reshape(m.size, m.batchMode)
end
Model.makeCleanModule['nn.Dropout'] = function (self, m)
   return nn.Dropout(m.p, not m.v2, m.inplace)
end
Model.makeCleanModule['nn.Linear'] = function (self, m)
   local new = nn.Linear(m.weight:size(2), m.weight:size(1), m.bias)
   new.weight:copy(m.weight)
   new.bias:copy(m.bias)
   return new
end
Model.makeCleanModule['nn.SpatialConvolution'] = function (self, m)
   local new = nn.SpatialConvolution(
      m.nInputPlane, m.nOutputPlane, m.kW, m.kH, m.dW, m.dH, m.padW, m.padH)
   new.weight:copy(m.weight)
   new.bias:copy(m.bias)
   return new
end
Model.makeCleanModule['cudnn.SpatialConvolution'] =
   Model.makeCleanModule['nn.SpatialConvolution']
Model.makeCleanModule['nn.SpatialMaxPooling'] = function (self, m)
   return nn.SpatialMaxPooling(m.kW, m.kH, m.dW, m.dH, m.padW, m.padH)
end
Model.makeCleanModule['cudnn.SpatialMaxPooling'] =
   Model.makeCleanModule['nn.SpatialMaxPooling']
Model.makeCleanModule['nn.TemporalConvolutionMM'] = function (self, m)
   local new = nn.TemporalConvolutionMM(
      m.input_feature, m.output_feature, m.kernel, m.stride, m.pad)
   new.weight:copy(m.weight)
   new.bias:copy(m.bias)
   return new
end
Model.makeCleanModule['cudnn.TemporalConvolutionCudnn'] = function (self, m)
   local new = nn.TemporalConvolutionMM(
      m.nInputPlane, m.nOutputPlane, m.kW, m.dW, m.padW)
   new.weight:copy(m.weight)
   new.bias:copy(m.bias)
   return new
end
Model.makeCleanModule['nn.TemporalMaxPoolingMM'] = function (self, m)
   return nn.TemporalMaxPoolingMM(m.kW, m.dW)
end
Model.makeCleanModule['cudnn.TemporalMaxPoolingCudnn'] =
   Model.makeCleanModule['nn.TemporalMaxPoolingMM']

-- Make CUDA modules
Model.makeCudaModule = {}
Model.makeCudaModule['nn.LogSoftMax'] = function (self, m)
   if self.cudnn and cudnn.LogSoftMax then
      return cudnn.LogSoftMax()
   else
      return nn.LogSoftMax()
   end
end
Model.makeCudaModule['cudnn.LogSoftMax'] = Model.makeCudaModule['nn.LogSoftMax']
Model.makeCudaModule['nn.Threshold'] = function (self, m)
   return nn.Threshold(m.threshold, m.val, m.inplace)
end
Model.makeCudaModule['nn.Reshape'] = function (self, m)
   return nn.Reshape(m.size, m.batchMode)
end
Model.makeCudaModule['nn.Dropout'] = function (self, m)
   return nn.Dropout(m.p, not m.v2, m.inplace)
end
Model.makeCudaModule['nn.Linear'] = function (self, m)
   local new = nn.Linear(m.weight:size(2), m.weight:size(1), m.bias)
   new.weight:copy(m.weight)
   new.bias:copy(m.bias)
   return new
end
Model.makeCudaModule['nn.SpatialConvolution'] = function (self, m)
   local new
   if self.cudnn then
      new = cudnn.SpatialConvolution(
         m.nInputPlane, m.nOutputPlane, m.kW, m.kH, m.dW, m.dH, m.padW, m.padH)
   else
      new = nn.SpatialConvolution(
         m.nInputPlane, m.nOutputPlane, m.kW, m.kH, m.dW, m.dH, m.padW, m.padH)
   end
   new.weight:copy(m.weight)
   new.bias:copy(m.bias)
   return new
end
Model.makeCudaModule['cudnn.SpatialConvolution'] =
   Model.makeCudaModule['nn.SpatialConvolution']
Model.makeCudaModule['nn.SpatialMaxPooling'] = function (self, m)
   if self.cudnn then
      return cudnn.SpatialMaxPooling(m.kW, m.kH, m.dW, m.dH, m.padW, m.padH)
   else
      return nn.SpatialMaxPooling(m.kW, m.kH, m.dW, m.dH, m.padW, m.padH)
   end
end
Model.makeCudaModule['cudnn.SpatialMaxPooling'] =
   Model.makeCudaModule['nn.SpatialMaxPooling']
Model.makeCudaModule['nn.TemporalConvolutionMM'] = function (self, m)
   local new
   if self.cudnn then
      new = cudnn.TemporalConvolutionCudnn(
         m.input_feature, m.output_feature, m.kernel, m.stride, m.pad)
   else
      new = nn.TemporalConvolutionMM(
         m.input_feature, m.output_feature, m.kernel, m.stride, m.pad)
   end
   new.weight:copy(m.weight)
   new.bias:copy(m.bias)
   return new
end
Model.makeCudaModule['cudnn.TemporalConvolutionCudnn'] = function (self, m)
   local new
   if self.cudnn then
      new = cudnn.TemporalConvolutionCudnn(
         m.nInputPlane, m.nOutputPlane, m.kW, m.dW, m.padW)
   else
      new = nn.TemporalConvolutionMM(
         m.nInputPlane, m.nOutputPlane, m.kW, m.dW, m.padW)
   end
   new.weight:copy(m.weight)
   new.bias:copy(m.bias)
   return new
end
Model.makeCudaModule['nn.TemporalMaxPoolingMM'] = function (self, m)
   if self.cudnn then
      return cudnn.TemporalMaxPoolingCudnn(m.kW, m.dW)
   else
      return nn.TemporalMaxPoolingMM(m.kW, m.dW)
   end
end
Model.makeCudaModule['cudnn.TemporalMaxPoolingCudnn'] =
   Model.makeCudaModule['nn.TemporalMaxPoolingMM']

return Model

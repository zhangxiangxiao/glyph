--[[
Driver for GlyphNet training
Copyright 2016 Xiang Zhang
--]]

local class = require('pl.class')
local math = require('math')
local nn = require('nn')
local os = require('os')
local paths = require('paths')
local torch = require('torch')

local Data = require('data')
local Model = require('model')
local Train = require('train')
local Test = require('test')

local Driver = class()

-- Constructor for driver
-- options: configuration table for other classes
-- config: configuration table for driver
--   .type: tensor type to do computation
--   .device: device id for CUDA. Only valid for .type = 'torch.CudaTensor'
--   .loss: the loss class to be used
--   .variation: the variation of the model
--   .steps: number of steps for each epoch
--   .epoches: number of epoches
--   .rate: initial learning rate
--   .schedule: rate change schedule
--   .interval: print time interval
--   .location: save location
--   .plot: whether to plot the result
--   .visualize: whether to visualize the models
--   .debug: whether to do debugging
--   .resume: whether to do resumption
function Driver:_init(options, config)
   local config = config or {}
   self.type = config.type or 'torch.DoubleTensor'
   self.device = config.device or 1
   self.loss = config.loss or 'nn.ClassNLLCriterion'
   self.variation = config.variation or 'large'
   self.steps = config.steps or 100000
   self.epoches = config.epoches or 100
   self.rate = config.rate or 1e-3
   self.schedule = config.schedule or 8
   self.interval = config.interval or 5
   self.location = config.location or '.'
   self.plot = config.plot
   self.visualize = config.visualize
   self.debug = config.debug
   self.resume = config.resume
   self.options = options

   -- Update the rates for training
   local rates = {}
   for i, v in pairs(self.options.train.rates) do
      rates[(i - 1) * self.steps * self.schedule + 1] = v * self.rate
      self.options.train.rates = rates
   end

   -- CUDA settings
   if self.type == 'torch.CudaTensor' then
      local cutorch = require('cutorch')
      print('Driver setting device to '..self.device)
      cutorch.setDevice(self.device)
   end

   -- Initialize random seed
   math.randomseed(os.time())
   torch.manualSeed(os.time())

   -- Handle model variation
   self:initVariation()

   -- Load data
   print('Driver loading training data')
   self.train_data = Data(self.options.train_data)
   print('Driver loading testing data')
   self.test_data = Data(self.options.test_data)

   -- Handle final output number of classes. Assuming last module is nn.Linear.
   local num_class = self.train_data:getClasses()
   for i = #self.options.model.temporal, 1, -1 do
      if self.options.model.temporal[i].name == 'nn.Linear' then
         print('Driver adjusting number of classes in model to '..num_class)
         self.options.model.temporal[i].outputSize = num_class
         break
      end
   end

   -- Handle resumption
   if self.resume then
      local record_file = paths.concat(self.location, 'record.t7b')
      print('Driver loading resumption from '..record_file)
      self.record = torch.load(record_file)

      local model_file = paths.concat(
         self.location, 'model_'..#self.record..'.t7b')
      print('Driver loading model from '..model_file)
      self.options.model.file = model_file
      self.model = Model(self.options.model)

      local state_file = paths.concat(
         self.location, 'state_'..#self.record..'.t7b')
      print('Driver loading training state from '..state_file)
      self.options.train.state = torch.load(state_file)
 
      print('Driver setting train step to '..(#self.record * self.steps))
      self.options.train.step = #self.record * self.steps

      for i = 1, #self.record do
         self:printResult(i)
      end
      if self.plot then
         self:plotRecord()
      end
   else
      self.record = {}
      print('Driver loading model')
      self.model = Model(self.options.model)
   end

   print('Driver setting model type to '..self.type)
   self.model:type(self.type)
   print('Driver loading trainer')
   self.trainer_loss = nn[self.loss:sub(4)]()
   self.trainer_loss:type(self.type)
   self.trainer = Train(
      self.train_data, self.model, self.trainer_loss, self.options.train)
   print('Driver loading tester for training data')
   self.train_tester_loss = nn[self.loss:sub(4)]()
   self.train_tester_loss:type(self.type)
   self.train_tester = Test(
      self.train_data, self.model, self.train_tester_loss, self.options.test)
   print('Driver loading tester for testing data')
   self.test_tester_loss = nn[self.loss:sub(4)]()
   self.test_tester_loss:type(self.type)
   self.test_tester = Test(
      self.test_data, self.model, self.test_tester_loss, self.options.test)

   if self.visualize then
      self:visualizeModel()
   end

   self.time = os.time()
end

-- Initialize variation
function Driver:initVariation()
   print('Driver using model variation '..self.variation)
   self.options.model.spatial = self.options.variation[self.variation].spatial
   self.options.model.temporal = self.options.variation[self.variation].temporal

   print('Driver adjusting data length to '..
            self.options.variation[self.variation].length)
   self.options.train_data.length =
      self.options.variation[self.variation].length
   self.options.test_data.length =
      self.options.variation[self.variation].length
end

-- Run the training process
function Driver:run()
   local begin_epoch = #self.record + 1
   local end_epoch = #self.record + self.epoches
   for i = begin_epoch, end_epoch do
      print('Driver setting model to training mode')
      self.model:setModeTrain()
      print('Driver training for epoch '..i)
      self.trainer:run(
         self.steps, function(train, step) self:logTrain(train, step) end)
      if self.visualize then
         self:visualizeModel()
      end

      print('Driver setting model to testing mode')
      self.model:setModeTest()
      print('Driver testing on training data for epoch '..i)
      self.train_tester:run(function(test, step) self:logTest(test, step) end)
      print('Driver testing on testing data for epoch '..i)

      self.test_tester:run(function(test, step) self:logTest(test, step) end)
      print('Driver saving for epoch '..i)
      self:save()
      self:printResult()
      if self.plot then
         self:plotRecord()
      end
   end
end

-- Save the record and the model
function Driver:save()
   local epoch = epoch or #self.record + 1

   -- Make a backup for the record
   print('Driver backing up record.t7b')
   local record_file = paths.concat(self.location, 'record.t7b')
   os.rename(record_file, record_file..'.backup')

   -- Save the new record
   print('Driver saving new records to '..record_file)
   self.record[epoch] = {
      train_loss = self.train_tester.total_objective,
      test_loss = self.test_tester.total_objective,
      train_error = self.train_tester.total_error,
      test_error = self.test_tester.total_error
   }
   torch.save(record_file, self.record)

   -- Save the model
   local model_file = paths.concat(self.location, 'model_'..epoch..'.t7b')
   print('Driver saving model to '..model_file)
   self.model:save(model_file)

   -- Save the training state
   local state_file = paths.concat(self.location, 'state_'..epoch..'.t7b')
   print('Driver saving training state to '..state_file)
   torch.save(state_file, self.trainer.state:type(torch.getdefaulttensortype()))
end

-- Print current result
function Driver:printResult(epoch)
   local epoch = epoch or #self.record
   print('Driver epoch = '..epoch..
            ', train_error = '..self.record[epoch].train_error..
            ', test_error = '..self.record[epoch].test_error..
            ', train_loss = '..self.record[epoch].train_loss..
            ', test_loss = '..self.record[epoch].test_loss)
end

-- Plot the record
function Driver:plotRecord()
   require('gnuplot')
   self.error_figure = self.error_figure or gnuplot.figure()
   self.loss_figure = self.loss_figure or gnuplot.figure()

   local epoch = torch.linspace(1, #self.record, #self.record)
   local train_error = torch.Tensor(epoch:size())
   local test_error = torch.Tensor(epoch:size())
   local train_loss = torch.Tensor(epoch:size())
   local test_loss = torch.Tensor(epoch:size())
   for i = 1, #self.record do
      train_error[i] = self.record[i].train_error
      test_error[i] = self.record[i].test_error
      train_loss[i] = self.record[i].train_loss
      test_loss[i] = self.record[i].test_loss
   end

   gnuplot.figure(self.error_figure)
   gnuplot.plot({'Training error', epoch, train_error},
                {'Testing error', epoch, test_error})
   gnuplot.title('Training and testing error')
   gnuplot.figure(self.loss_figure)
   gnuplot.plot({'Training loss', epoch, train_loss},
                {'Testing loss', epoch, test_loss})
   gnuplot.title('Training and testing loss')
end

-- Visualize the model
function Driver:visualizeModel()
   local Visualizer = require('visualizer')
   self.options.visualizer.title = 'Spatial model'
   self.spatial_visualizer = self.spatial_visualizer or
      Visualizer(self.options.visualizer)
   self.options.visualizer.title = 'Temporal model'
   self.temporal_visualizer = self.temporal_visualizer or
      Visualizer(self.options.visualizer)
   self.options.visualizer.title = nil

   self.spatial_visualizer:drawSequential(self.model.spatial)
   self.temporal_visualizer:drawSequential(self.model.temporal)
end

-- Log training
function Driver:logTrain(train, step)
   -- If it is not time to log, return
   if os.difftime(os.time(), self.time) < self.interval then return end

   local message = 'Train step = '..train.step..
      ', rate = '..string.format('%.2e', train.rate)..
      ', error = '..string.format('%.2e', train.error)..
      ', loss = '..string.format('%.2e', train.objective)..
      ', data = '..string.format('%.2e', train.time.data)..
      ', forward = '..string.format('%.2e', train.time.forward)..
      ', backward = '..string.format('%.2e', train.time.backward)..
      ', update = '..string.format('%.2e', train.time.update)

   if self.debug then
      message = message..
         ', input = ['..string.format("%.2e",train.input:min())..
         ' '..string.format("%.2e",train.input:max())..
         ' '..string.format("%.2e",train.input:mean())..
         ' '..string.format("%.2e",train.input:std())..']'..
         ', params = ['..string.format("%.2e",train.params:min())..
         ' '..string.format("%.2e",train.params:max())..
         ' '..string.format("%.2e",train.params:mean())..
         ' '..string.format("%.2e",train.params:std())..']'..
         ', grads = ['..string.format("%.2e",train.grads:min())..
         ' '..string.format("%.2e",train.grads:max())..
         ' '..string.format("%.2e",train.grads:mean())..
         ' '..string.format("%.2e",train.grads:std())..']'..
         ', state = ['..string.format("%.2e",train.state:min())..
         ' '..string.format("%.2e",train.state:max())..
         ' '..string.format("%.2e",train.state:mean())..
         ' '..string.format("%.2e",train.state:std())..']'

      if self.visualize then
         self:visualizeModel()
      end
   end

   print(message)
   self.time = os.time()
end

-- Log testing
function Driver:logTest(test)
   -- If it not time to log, return
   if os.difftime(os.time(), self.time) < self.interval then return end

   local message = 'Test count = '..test.total_count..
      ', error = '..string.format('%.2e', test.error)..
      ', loss = '..string.format('%.2e', test.objective)..
      ', total_error = '..string.format('%.2e', test.total_error)..
      ', total_loss = '..string.format('%.2e', test.total_objective)..
      ', data = '..string.format('%.2e', test.time.data)..
      ', forward = '..string.format('%.2e', test.time.forward)..
      ', update = '..string.format('%.2e', test.time.update)

   if self.debug then
      message = message..
         ', input = ['..string.format("%.2e",test.input:min())..
         ' '..string.format("%.2e",test.input:max())..
         ' '..string.format("%.2e",test.input:mean())..
         ' '..string.format("%.2e",test.input:std())..']'
   end

   print(message)
   self.time = os.time()
end

return Driver

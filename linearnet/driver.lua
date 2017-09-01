--[[
Driver for LinearNet using HogWILD!
Copyright 2015 Xiang Zhang
--]]

local class = require('pl.class')
local math = require('math')
local os = require('os')
local paths = require('paths')
local threads = require('threads')
local torch = require('torch')

local Data = require('data')
local Model = require('model')
local Qeueu = require('queue')
local Train = require('train')
local Test = require('test')

-- Library configurations
threads.serialization('threads.sharedserialize')

local Driver = class()

-- Constructor for driver
-- options: configuration table for others
-- config: configuration table
--    .loss: the loss used for classification task
--    .threads: number of threads
--    .buffer: buffer size for RPC queues
--    .steps: steps for each training run
--    .epoches: number of testing epoches before stopping
--    .interval: print time interval
--    .location: save location
--    .initialization: initialization parameter for model
--    .plot: whether to plot the output
--    .debug: whether to debug
--    .resume: whether to resume
function Driver:_init(options, config)
   local config = config or {}
   self.loss = config.loss or 'nn.ClassNLLCriterion'
   self.threads = config.threads or 10
   self.buffer = config.buffer or 100
   self.steps = config.steps or 100000
   self.epoches = config.epoches or 1000
   self.interval = config.interval or 5
   self.location = config.location or '.'
   self.initialization = config.initialization or 1e-2
   self.plot = config.plot
   self.debug = config.debug
   self.resume = config.resume
   self.options = options or {}
   self.config = config

   math.randomseed(os.time())
   torch.manualSeed(os.time())

   print('Driver loading training data')
   self.train_data = Data(self.options.train_data)
   print('Driver loading testing data')
   self.test_data = Data(self.options.test_data)
   self.options.model.dimension = self.train_data:getClasses()
   print('Driver changed model output dimension to '..
            self.options.model.dimension)

   if self.resume then
      local record_file = paths.concat(self.location, 'record.t7b')
      print('Driver loading resumption record from '..record_file)
      self.record = torch.load(record_file)
      local model_file = paths.concat(
         self.location, 'model_'..#self.record..'.t7b')
      print('Driver loading model from '..model_file)
      self.model = Model(self.options.model)
      self.model:load(model_file)
      if self.record[#self.record].progress then
         if self.record[#self.record].progress:size(1) == self.threads then
            self.progress = self.record[#self.record].progress:clone()
         else
            print('Driver resumption number of threads change.')
            self.progress = torch.LongTensor(self.threads):zero()
            local total = self.record[#self.record].progress:sum()
            while self.progress:sum() < total do
               local thread = math.random(self.threads)
               self.progress[thread] = self.progress[thread] + self.steps 
            end
         end
      else
         print('Driver resumption progress vector not found')
         self.progress = torch.LongTensor(self.threads):zero()
      end
      print('Driver progress = '..self.progress:sum())
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
      print('Driver initializing model')
      self.model:reset(self.initialization)
      self.progress = torch.LongTensor(self.threads):zero()
      if self.plot then
         require('gnuplot')
      end
   end

   print('Driver loading tester for training data')
   self.train_test = Test(
      self.train_data, self.model, nn[self.loss:sub(4)](), self.options.test)
   print('Driver loading tester for testing data')
   self.test_test = Test(
      self.test_data, self.model, nn[self.loss:sub(4)](), self.options.test)

   print('Driver building RPC queues')
   self.master_queue = Queue(self.buffer)
   self.slave_queues = {}
   for i = 1, self.threads do
      self.slave_queues[i] = Queue(self.buffer)
   end

   print('Driver creating thread block')
   local init_thread = self:initThread()
   self.block = threads.Threads(self.threads, init_thread)
   self.block:specific(true)

   self.time = os.time()
   self.step = self.progress:sum()
end

-- Run the training process
function Driver:run()
   self:deployThreads()

   local begin_epoch = #self.record + 1
   local end_epoch = #self.record + self.epoches
   for i = begin_epoch, end_epoch do
      print('Driver testing on training data for epoch '..i)
      self.train_test:run(function (test, step) self:logTest(test, step) end)
      print('Driver testing on testing data for epoch '..i)
      self.test_test:run(function (test, step) self:logTest(test, step) end)
      self:save()
      self:printResult()
      if self.plot then
         self:plotRecord()
      end
   end

   for i = 1, self.threads do
      print('Driver sending RPC to exit thread '..i)
      self.slave_queues[i]:push{func = 'exit', arg = {}}
   end

   self.block:synchronize()
   self.block:terminate()
end

-- Deploy threads in sequential order to prevent io and memory jam
function Driver:deployThreads()
   for i = 1, self.threads do
      print('Driver deploying job for threads '..i)
      local thread_job = self:threadJob(i)
      self.block:addjob(i, thread_job)
      local rpc = self.master_queue:pop()
      while rpc.func ~= 'notifyDeploy' do
         self[rpc.func](self, unpack(rpc.arg))
         rpc = self.master_queue:pop()
      end
      print('Driver rpc = notifyDeploy, thread = '..rpc.arg[1])
   end
end

-- Thread initialization callback
function Driver:initThread()
   return function ()
      local math = require('math')
      local nn = require('nn')
      local os = require('os')
      local torch = require('torch')

      local Queue = require('queue')

      math.randomseed(os.time() + __threadid)
      torch.manualSeed(os.time() + __threadid)
   end
end

-- Thread job callback
function Driver:threadJob(id)
   local options = self.options
   local steps = self.steps
   local data_table = self.train_data:getTable()
   local modules = self.model:getModules()
   local loss = self.loss
   local master_queue = self.master_queue
   local slave_queue = self.slave_queues[id]
   local progress = self.progress[id]
   return function()
      local os = require('os')
      local nn = require('nn')
      local torch = require('torch')

      local Data = require('data')
      local Model = require('model')
      local Train = require('train')

      local train_data = Data(options.train_data, data_table)
      local model = Model(options.model, modules)

      options.train.step = progress
      local train = Train(train_data, model, nn[loss:sub(4)](), options.train)
      master_queue:push{func = 'notifyDeploy', arg = {__threadid}}

      local exit = false
      while not exit do
         train:run(steps)
         -- Tell main thread to update progress
         master_queue:push{
            func = 'updateProgress',
            arg = {__threadid, train.step, train.objective}}
         -- Handle RPC requests from main thread
         local rpc = slave_queue:pop_async()
         while rpc do
            if rpc.func == 'exit' then
               exit = true
            end
            rpc = slave_queue:pop_async()
         end
      end
   end
end

-- Update progress
function Driver:updateProgress(thread, step, objective)
   self.progress[thread] = step
   print('Driver rpc = updateProgress, thread = '..thread..', objective = '..
            objective..', progress = '..self.progress[thread]..', total = '..
            self.progress:sum())
end

-- Log for testing
function Driver:logTest(test, step)
   if os.difftime(os.time(), self.time) >= self.interval then
      local message = 'Test step = '..step..
         ', total_error = '..test.total_error..
         ', total_objective = '..test.total_objective..
         ', label = '..test.label[1]..
         ', decision = '..test.decision[1]
      if self.debug then
         local weight = {
            weight = test.model.linear.weight, bias = test.model.linear.bias}
         for key, w in pairs(weight) do
            message = message..', '..key..':mean() = '..w:mean()..', '..
               key..':std() = '..w:std()
         end
      end
      print(message)

      -- Handle rpc
      local rpc = self.master_queue:pop_async()
      while rpc do
         self[rpc.func](self, unpack(rpc.arg))
         rpc = self.master_queue:pop_async()
      end
      self.time = os.time()
   end
end

-- Save for model
function Driver:save(epoch)
   local epoch = epoch or #self.record + 1

   -- Make a backup for the record
   print('Driver backing up record.t7b')
   local record_file = paths.concat(self.location, 'record.t7b')
   os.rename(record_file, record_file..'.backup')

   -- Save the new record
   print('Driver saving new records to '..record_file)
   self.record[epoch] = {
      train_loss = self.train_test.total_objective,
      test_loss = self.test_test.total_objective,
      train_error = self.train_test.total_error,
      test_error = self.test_test.total_error,
      progress = self.progress:clone()
   }
   torch.save(record_file, self.record)

   -- Save the model
   local model_file = paths.concat(self.location, 'model_'..epoch..'.t7b')
   print('Driver saving model to '..model_file)
   self.model:save(model_file)
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

return Driver

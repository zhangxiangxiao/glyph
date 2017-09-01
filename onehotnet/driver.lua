--[[
Driver for OnehotNet training
Copyright 2016 Xiang Zhang
--]]

local class = require('pl.class')

local parent = require('glyphnet/driver')
local Driver = class(parent)

-- Initialize variation
function Driver:initVariation()
   print('Driver using model variation '..self.variation)
   self.options.model.onehot =
      self.options.variation[self.variation].onehot
   self.options.model.temporal = self.options.variation[self.variation].temporal

   print('Driver adjusting data length to '..
            self.options.variation[self.variation].length)
   self.options.train_data.length =
      self.options.variation[self.variation].length
   self.options.test_data.length =
      self.options.variation[self.variation].length
end

-- Visualize the model
function Driver:visualizeModel()
   local Visualizer = require('visualizer')
   self.options.visualizer.title = 'Onehot model'
   self.onehot_visualizer = self.onehot_visualizer or
      Visualizer(self.options.visualizer)
   self.options.visualizer.title = 'Temporal model'
   self.temporal_visualizer = self.temporal_visualizer or
      Visualizer(self.options.visualizer)
   self.options.visualizer.title = nil

   self.onehot_visualizer:drawSequential(self.model.onehot)
   self.temporal_visualizer:drawSequential(self.model.temporal)
end

return Driver

--[[
Driver for EmbedNet training
Copyright 2016 Xiang Zhang
--]]

local class = require('pl.class')

local parent = require('glyphnet/driver')
local Driver = class(parent)

-- Initialize variation
function Driver:initVariation()
   print('Driver using model variation '..self.variation)
   self.options.model.embedding =
      self.options.variation[self.variation].embedding
   self.options.model.temporal = self.options.variation[self.variation].temporal

   print('Driver adjusting data length to '..
            self.options.variation[self.variation].length)
   self.options.train_data.length =
      self.options.variation[self.variation].length
   self.options.test_data.length =
      self.options.variation[self.variation].length

   self.dimension = self.options.driver.dimension
   print('Driver adjusting data index dimension to '..self.dimension)
   self.options.model.embedding[1].nIndex = self.dimension
   self.options.model.embedding[1].paddingValue =
      self.options.train_data.replace
end

-- Visualize the model
function Driver:visualizeModel()
   local Visualizer = require('visualizer')
   self.options.visualizer.title = 'Embedding model'
   self.embedding_visualizer = self.embedding_visualizer or
      Visualizer(self.options.visualizer)
   self.options.visualizer.title = 'Temporal model'
   self.temporal_visualizer = self.temporal_visualizer or
      Visualizer(self.options.visualizer)
   self.options.visualizer.title = nil

   self.embedding_visualizer:drawSequential(self.model.embedding)
   self.temporal_visualizer:drawSequential(self.model.temporal)
end

return Driver

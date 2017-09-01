--[[
Visualization module for EmbedNet
Copyright 2016 Xiang Zhang
--]]

local class = require('pl.class')

local parent = require('glyphnet/visualizer')
local Visualizer = class(parent)

Visualizer.drawModule['nn.LookupTable'] = Visualizer.drawModule['nn.Linear']

return Visualizer

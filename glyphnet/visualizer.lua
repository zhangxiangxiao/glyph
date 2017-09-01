--[[
Visualization module for glyphnet
Copyright 2016 Xiang Zhang
--]]

local class = require('pl.class')
local torch = require('torch')

local Scroll = require('scroll')

local Visualizer = class()

-- Constructor
--  config: configuration table
--    .width: (optional) width of scrollable window
--    .scale: (optional) scale of visualizing weights
--    .title: (optional) title of the scrollable window
--    .height: (optional) maximum height of visualization for a module
function Visualizer:_init(config)
   local config = config or {}
   local config = config or {}
   self.width = config.width or 800
   self.scale = config.scale or 4
   self.title = config.title or "Visualizer"
   self.height = config.height or 64
   self.win = Scroll(self.width, self.title)
end

-- Save wrapper
function Visualizer:save(...)
   return self.win:save(...)
end

-- Visualize the weights of a sequential model
-- model: the sequential model
function Visualizer:drawSequential(model)
   self.win:clear()
   for i, m in ipairs(model.modules) do
      self.win:drawText(tostring(i)..": "..tostring(m))
      if self.drawModule[torch.type(m)] then
         self.drawModule[torch.type(m)](self, m)
      end
   end
end

-- Draw an image with height hints
function Visualizer:drawImage(im, y_zero, max, min)
   local win = self.win
   local y = win:hintImageHeight(im, self.scale)
   if y - y_zero > self.height then
      return false
   end
   local max = max or im:max()
   local min = min or im:min()
   local normalized = torch.Tensor(im:size()):copy(im):add(-min)
   if max - min > 0 then
      normalized:div(max - min)
   end
   win:drawImage(normalized, self.scale)
   return true
end

-- A table for reading modules
Visualizer.drawModule = {}
Visualizer.drawModule['nn.Linear'] = function (self, m)
   local weight = m.weight
   local y_zero = self.win.y

   for i = 1, m.weight:size(1) do
      local w = weight[i]:view(1, weight:size(2))
      if not self:drawImage(w, y_zero) then
         return
      end
   end

   self:drawImage(m.bias:view(1, m.bias:size(1)), y_zero)
end
Visualizer.drawModule['nn.SpatialConvolution'] = function (self, m)
   local weight = m.weight:view(m.nOutputPlane, m.nInputPlane, m.kH, m.kW)
   local height = m.kH
   local width = m.kW
   local y_zero = self.win.y
   local max = weight:max()
   local min = weight:min()
   
   if m.nInputPlane == 3 then
      for i = 1, m.nOutputPlane do
         local w = weight[i]
         if not self:drawImage(w, y_zero, max, min) then
            return
         end
      end
   else
      for i = 1, m.nOutputPlane do
         for j = 1, m.nInputPlane do
            local w = weight[i][j]
            if not self:drawImage(w, y_zero, max, min) then
               return
            end
         end
      end
   end

   self:drawImage(m.bias:view(1, m.nOutputPlane), y_zero)
end
Visualizer.drawModule['nn.SpatialConvolutionMM'] =
   Visualizer.drawModule['nn.SpatialConvolution']
Visualizer.drawModule['cudnn.SpatialConvolution'] =
   Visualizer.drawModule['nn.SpatialConvolution']
Visualizer.drawModule['nn.TemporalConvolutionMM'] = function (self, m)
   local weight = m.weight:view(m.output_feature, m.input_feature, m.kernel)
   local y_zero = self.win.y
   local max = weight:max()
   local min = weight:min()

   for i = 1, m.output_feature do
      local w = weight[i]:transpose(2, 1)
      if not self:drawImage(w, y_zero, max, min) then
         return
      end
   end
end
Visualizer.drawModule['cudnn.TemporalConvolutionCudnn'] = function (self, m)
   local weight = m.weight:view(m.nOutputPlane, m.nInputPlane, m.kW)
   local y_zero = self.win.y
   local max = weight:max()
   local min = weight:min()

   for i = 1, m.nOutputPlane do
      local w = weight[i]:transpose(2, 1)
      if not self:drawImage(w, y_zero, max, min) then
         return
      end
   end
end

return Visualizer

--[[
Additional modules for GlyphNet
Copyright 2016 Xiang Zhang
--]]

local status, cudnn = pcall(require, 'cudnn')
local nn = require('nn')

-- nn.TemporalConvolutionMM
if not nn.TemporalConvolutionMM then
   dofile('modules/TemporalConvolutionMM.lua')
end

-- nn.TemporalMaxPoolingMM
if not nn.TemporalMaxPoolingMM then
   dofile('modules/TemporalMaxPoolingMM.lua')
end

-- cudnn.TemporalConvolutionCudnn
if status == true and not cudnn.TemporalMaxPoolingCudnn then
   dofile('modules/TemporalConvolutionCudnn.lua')
end

-- cudnn.TemporalMaxPoolingCudnn
if status == true and not cudnn.TemporalMaxPoolingCudnn then
   dofile('modules/TemporalMaxPoolingCudnn.lua')
end

return nn

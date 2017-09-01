--[[
Temporal max pooling module with data order consistent with MM
Copyright 2016 Xiang Zhang
--]]

local TemporalMaxPoolingMM, parent =
   torch.class('nn.TemporalMaxPoolingMM', 'nn.SpatialMaxPooling')

function TemporalMaxPoolingMM:__init(kW, dW)
   parent.__init(self, kW, 1, dW, 1)
end

function TemporalMaxPoolingMM:updateOutput(input)
   local input_view

   if input:dim() == 2 then
      input_view = input:view(input:size(1), 1, input:size(2))
   else
      input_view = input:view(input:size(1), input:size(2), 1, input:size(3))
   end

   local output = parent.updateOutput(self, input_view)

   if input:dim() == 2 then
      self.output = output:view(output:size(1), output:size(3))
   else
      self.output = output:view(output:size(1), output:size(2), output:size(4))
   end

   return self.output
end

function TemporalMaxPoolingMM:updateGradInput(input, grad_output)
   local input_view
   local grad_output_view
   if input:dim() == 2 then
      input_view = input:view(input:size(1), 1, input:size(2))
      grad_output_view = grad_output:view(
         grad_output:size(1), 1, grad_output:size(2))
      self.output = self.output:view(
         self.output:size(1), 1, self.output:size(2))
   else
      input_view = input:view(input:size(1), input:size(2), 1, input:size(3))
      grad_output_view = grad_output:view(
         grad_output:size(1), grad_output:size(2), 1, grad_output:size(3))
      self.output = self.output:view(
         self.output:size(1), self.output:size(2), 1, self.output:size(3))
   end

   local grad_input = parent.updateGradInput(self, input_view, grad_output_view)

   if input:dim() == 2 then
      self.output = self.output:view(
         self.utput:size(1), self.output:size(3))
      self.gradInput = grad_input:view(grad_input:size(1), grad_input:size(3))
   else
      self.output = self.output:view(
         self.output:size(1), self.output:size(2), self.output:size(4))
      self.gradInput = grad_input:view(
         grad_input:size(1), grad_input:size(2), grad_input:size(4))
   end

   return self.gradInput
end

function TemporalMaxPoolingMM:__tostring__()
   return string.format('%s(%d, %d)', torch.type(self), self.kW, self.dW)
end

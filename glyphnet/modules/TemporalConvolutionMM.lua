--[[
Temporal convolution module that supports padding
Copyright 2016 Xiang Zhang
--]]

local TemporalConvolutionMM, parent =
   torch.class('nn.TemporalConvolutionMM', 'nn.Module')

function TemporalConvolutionMM:__init(
      input_feature, output_feature, kernel, stride, pad)
   parent.__init(self)
   
   self.input_feature = input_feature
   self.output_feature = output_feature
   self.kernel = kernel
   self.stride = stride or 1
   self.pad = pad or 0
   
   self.weight = torch.Tensor(output_feature, input_feature, kernel)
   self.bias = torch.Tensor(output_feature)
   self.gradWeight = torch.Tensor(output_feature, input_feature, kernel)
   self.gradBias = torch.Tensor(output_feature)
   
   self.pad_cache = torch.Tensor()
   self.unfold_cache = torch.Tensor()
   self.interlace_cache = torch.Tensor()
   self.weight_cache = torch.Tensor(
      self.weight:size(2), self.weight:size(1), self.weight:size(3))
   self.reverse_index = torch.LongTensor(self.kernel)
   
   for i = 1, self.kernel do
      self.reverse_index[i] = self.kernel - i + 1
   end
   
   self:reset()
end

function TemporalConvolutionMM:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kernel * self.input_feature)
   end
   self.weight:uniform(-stdv, stdv)
   self.bias:uniform(-stdv, stdv)
end

function TemporalConvolutionMM:updateOutput(input)
   if input:dim() ~= 2 and input:dim() ~= 3 then
      error('Input dimension must be 2 or 3')
   end
   
   -- Create temporary input cache that is to be unfolded
   if input:dim() == 2 then
      self.pad_cache:resize(
         input:size(1), input:size(2) + 2 * self.pad):zero():narrow(
         2, self.pad + 1, input:size(2)):copy(input)
   else
      self.pad_cache:resize(
         input:size(1), input:size(2),
         input:size(3) + 2 * self.pad):zero():narrow(
         3, self.pad + 1, input:size(3)):copy(input)
   end
   
   -- Unfold the input cache
   local unfolded = self.pad_cache:unfold(
      self.pad_cache:dim(), self.kernel, self.stride):transpose(
      self.pad_cache:dim(), self.pad_cache:dim() + 1)
   self.unfold_cache:resizeAs(unfolded):copy(unfolded)
   
   -- Do matrix multiplication
   if input:dim() == 2 then
      self.output:resize(
         self.output_feature, self.unfold_cache:size(3)):copy(
         self.bias:view(-1, 1):expandAs(self.output))
      self.output:addmm(
         1, self.output, 1,
         self.weight:view(self.weight:size(1), -1),
         self.unfold_cache:view(-1, self.unfold_cache:size(3)))
   else
      self.output:resize(
         self.unfold_cache:size(1), self.output_feature,
         self.unfold_cache:size(4)):copy(
         self.bias:view(1, -1, 1):expandAs(self.output))
      local weight = self.weight:view(
         1, self.weight:size(1),
         self.weight:size(2) * self.weight:size(3)):expand(
         self.unfold_cache:size(1), self.weight:size(1),
         self.weight:size(2) * self.weight:size(3))
      self.output:baddbmm(
         1, self.output, 1, weight,
         self.unfold_cache:view(
            self.unfold_cache:size(1), -1, self.unfold_cache:size(4)))
   end
   
   return self.output
end

function TemporalConvolutionMM:updateGradInput(input, grad_output)
   -- Reverse the weight on the kernel dimension
   self.weight_cache:indexCopy(
      3, self.reverse_index, self.weight:transpose(1, 2))
   
   -- Resize the initialize the interlace cache
   if input:dim() == 2 then
      self.interlace_cache:resize(
         grad_output:size(1),
         self.stride * (grad_output:size(2) - 1) + 1):zero()
      self.interlace_cache:narrow(
         2, 1, self.interlace_cache:size(2) - 1):unfold(
         2, self.stride, self.stride):select(3, 1):copy(
         grad_output:narrow(2, 1, grad_output:size(2) - 1))
      self.interlace_cache:select(2, self.interlace_cache:size(2)):copy(
         grad_output:select(2, grad_output:size(2)))
   else
      self.interlace_cache:resize(
         grad_output:size(1), grad_output:size(2),
         self.stride * (grad_output:size(3) - 1) + 1):zero()
      self.interlace_cache:narrow(
         3, 1, self.interlace_cache:size(3) - 1):unfold(
         3, self.stride, self.stride):select(4, 1):copy(
         grad_output:narrow(3, 1, grad_output:size(3) - 1))
      self.interlace_cache:select(3, self.interlace_cache:size(3)):copy(
         grad_output:select(3, grad_output:size(3)))
   end
   
   -- Resize and initialize the padded cache
   if input:dim() == 2 then
      self.pad_cache:resize(
         grad_output:size(1), input:size(2) + self.kernel - 1)
      local length = math.min(
         self.pad_cache:size(2), self.interlace_cache:size(2))
      self.pad_cache:zero():narrow(
         2, (self.pad_cache:size(2) - length) / 2 + 1, length):copy(
         self.interlace_cache:narrow(
            2, (self.interlace_cache:size(2) - length) / 2 + 1, length))
   else
      self.pad_cache:resize(
         grad_output:size(1), grad_output:size(2),
         input:size(3) + self.kernel - 1)
      local length = math.min(
         self.pad_cache:size(3), self.interlace_cache:size(3))
      self.pad_cache:zero():narrow(
         3, (self.pad_cache:size(3) - length) / 2 + 1, length):copy(
         self.interlace_cache:narrow(
            3, (self.interlace_cache:size(3) - length) / 2 + 1, length))
   end
   
   -- Unfold the output cache
   local unfolded = self.pad_cache:unfold(
      self.pad_cache:dim(), self.kernel, 1):transpose(
      self.pad_cache:dim(), self.pad_cache:dim() + 1)
   self.unfold_cache:resizeAs(unfolded):copy(unfolded)
   
   -- Do matrix multiplication
   self.gradInput:resizeAs(input):zero()
   if input:dim() == 2 then
      self.gradInput:addmm(
         1, self.gradInput, 1,
         self.weight_cache:view(self.weight:size(2), -1),
         self.unfold_cache:view(-1, self.unfold_cache:size(3)))
   else
      local weight = self.weight_cache:view(
         1, self.weight:size(2),
         self.weight:size(1) * self.weight:size(3)):expand(
         unfolded:size(1), self.weight:size(2),
         self.weight:size(1) * self.weight:size(3))
      self.gradInput:baddbmm(
         1, self.gradInput, 1, weight,
         self.unfold_cache:view(
            self.unfold_cache:size(1), -1, self.unfold_cache:size(4)))
   end
   
   return self.gradInput
end

function TemporalConvolutionMM:accGradParameters(input, grad_output, scale)
   local scale = scale or 1
   
   -- Create temporary input cache that is to be unfolded
   if input:dim() == 2 then
      self.pad_cache:resize(
         input:size(1), input:size(2) + 2 * self.pad):zero():narrow(
         2, self.pad + 1, input:size(2)):copy(input)
   else
      self.pad_cache:resize(
         input:size(1), input:size(2),
         input:size(3) + 2 * self.pad):zero():narrow(
         3, self.pad + 1, input:size(3)):copy(input)
   end

   -- Unfold the input cache
   local unfolded = self.pad_cache:unfold(
      self.pad_cache:dim(), self.kernel, self.stride):transpose(
      self.pad_cache:dim() - 1, self.pad_cache:dim())
   self.unfold_cache:resizeAs(unfolded):copy(unfolded)

   -- Do matrix multiplication
   local grad_weight = self.gradWeight:view(self.weight:size(1), -1)
   if input:dim() == 2 then
      grad_weight:addmm(
         1, grad_weight, scale, grad_output,
         self.unfold_cache:view(unfolded:size(1), -1))
      self.gradBias:add(scale, grad_output:sum(2))
   else
      if grad_weight.addbmm then
         grad_weight:addbmm(
            1, grad_weight, scale, grad_output,
            self.unfold_cache:view(
               self.unfold_cache:size(1), self.unfold_cache:size(2), -1))
      else
         for i = 1, grad_output:size(1) do
            grad_weight:addmm(
               1, grad_weight, scale, grad_output:select(1, i),
               self.unfold_cache:select(1, i):view(
                  self.unfold_cache:size(2), -1))
         end
      end
      self.gradBias:add(scale, grad_output:sum(3):sum(1))
   end
end

TemporalConvolutionMM.sharedAccUpdateGradParameters =
   TemporalConvolutionMM.accUpdateGradParameters

function TemporalConvolutionMM:__tostring__()
   return string.format(
      '%s(%d -> %d, %d, %d, %d)', torch.type(self), self.input_feature,
      self.output_feature, self.kernel, self.stride, self.pad)
end

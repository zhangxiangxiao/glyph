--[[
Unit test for modules
Copyright 2016 Xiang Zhang
--]]

local nn = require('modules')

local cunn = require('cunn')
local cutorch = require('cutorch')
local torch = require('torch')

-- A Logic Named Joe
local joe = {}

function joe.main()
   if joe.init then
      print('Initializing testing environment')
      joe.init(joe)
   end
   for name, func in pairs(joe) do
      if type(name) == 'string' and type(func) == 'function'
      and name:match('[%g]+Test') then
         print('\nExecuting '..name)
         func(joe)
      end
   end
end

function joe:init()
   local device = 1
   cutorch.setDevice(device)
   print('Device set to '..device)
   self.jacobian = nn.Jacobian
end

function joe:noBatchCPU(kernel, stride, pad)
   local input_feature = 2
   local output_feature = 4
   local kernel = kernel or 3
   local stride = stride or 1
   local pad = pad or 0
   local temporal = nn.TemporalConvolutionMM(
      input_feature, output_feature, kernel, stride, pad)
   print('Created module: '..tostring(temporal))
   temporal.gradWeight:zero()
   temporal.gradBias:zero()
   
   local output_length = 16
   local input_length = (output_length - 1) * stride + kernel - 2 * pad
   
   local input = torch.rand(input_feature, input_length)
   print('Input size:')
   print(input:size())
   
   print('Executing forward propagation')
   local output = temporal:forward(input)
   print('Output size: ')
   print(output:size())
   
   local input_pad = torch.zeros(input_feature, input_length + 2 * pad)
   input_pad:narrow(2, pad + 1, input_length):copy(input)
   for i = 1, output:size(2) do
      local input_begin = (i - 1) * stride + 1
      local input_chunk = input_pad:narrow(
	 2, input_begin, kernel):contiguous():view(
	 1, input_feature, kernel):expand(output_feature, input_feature, kernel)
      local output_slice = torch.cmul(
	 temporal.weight, input_chunk):sum(3):sum(2):squeeze()
      output_slice:add(1, temporal.bias:viewAs(output_slice))
      print('Error of output slice '..i..': '..
	       output_slice:add(-1, output:select(2, i)):abs():mean())
   end
   
   local grad_output = torch.rand(output:size())
   print('Executing backward propagation')
   local grad_input = temporal:backward(input, grad_output)
   print('Input gradient size: ')
   print(grad_input:size())
   
   local grad_output_pad = torch.Tensor(
      output_feature, input_length + kernel - 1):zero()
   local interlace_length = stride * (grad_output:size(2) - 1) + 1
   local interlace_shift = (grad_output_pad:size(2) - interlace_length) / 2
   for i = 1, grad_output:size(2) do
      local grad_output_pad_begin = (i - 1) * stride + 1 + interlace_shift
      if grad_output_pad_begin >= 1
	 and grad_output_pad_begin <= grad_output_pad:size(2) then
	 grad_output_pad:select(2, grad_output_pad_begin):copy(
	    grad_output:select(2, i))
      end
   end
   local weight_reverse = torch.Tensor(temporal.weight:size())
   local weight_index = torch.LongTensor(kernel)
   for i = 1, weight_index:size(1) do
      weight_index[i] = kernel - i + 1
   end
   weight_reverse:indexCopy(3, weight_index, temporal.weight)
   for i = 1, grad_input:size(2) do
      local grad_output_pad_begin = i
      local grad_output_pad_chunk = grad_output_pad:narrow(
	 2, grad_output_pad_begin, kernel):contiguous():view(
	 output_feature, 1, kernel):expand(
	 output_feature, input_feature, kernel)
      local grad_input_slice = torch.cmul(
	 weight_reverse, grad_output_pad_chunk):sum(3):sum(1):squeeze()
      print('Error of input gradient slice '..i..': '..
	       grad_input_slice:add(-1, grad_input:select(2, i)):abs():mean())
   end
   
   local input_unfold = input_pad:unfold(2, kernel, stride)
   for i = 1, temporal.weight:size(3) do
      local grad_weight_slice = torch.mm(
	 grad_output, input_unfold:select(3, i):transpose(1, 2))
      print('Error of weight gradient slice '..i..': '..grad_weight_slice:add(
		  -1, temporal.gradWeight:select(3, i)):abs():mean())
   end
   
   local grad_bias = grad_output:sum(2)
   print('Error of bias gradient: '..grad_bias:add(
	       -1, temporal.gradBias):abs():mean())
   
   local jacobian = self.jacobian
   local err = jacobian.testJacobian(temporal, input)
   print('Error of jacobian test: '..err)
   local err = jacobian.testJacobianParameters(
      temporal, input, temporal.weight, temporal.gradWeight)
   print('Error of jacobian test for weight: '..err)
   local err = jacobian.testJacobianParameters(
      temporal, input, temporal.bias, temporal.gradBias)
   print('Error of jacobian test for bias: '..err)
   local err = jacobian.testJacobianUpdateParameters(
      temporal, input, temporal.weight)
   print('Error of jacobian test for weight update: '..err)
   local err = jacobian.testJacobianUpdateParameters(
      temporal, input, temporal.bias)
   print('Error of jacobian test for bias update: '..err)
   for t,err in pairs(
      jacobian.testAllUpdate(temporal, input, 'weight', 'gradWeight')) do
      print('Error of jacobian test for '..t..' all update: '..err)
   end
   for t,err in pairs(
      jacobian.testAllUpdate(temporal, input, 'bias', 'gradBias')) do
      print('Error of jacobian test for '..t..' all update: '..err)
   end
end

function joe:noBatchCPUTest()
   for _, kernel in ipairs({3, 5}) do
      for _, stride in ipairs({1, 2, 3, 5}) do
	 for _, pad in ipairs({0, 1, 2, 3, 5}) do
	    self:noBatchCPU(kernel, stride, pad)
	 end
      end
   end
end

function joe:batchCPU(kernel, stride, pad)
   local batch = 4
   local input_feature = 2
   local output_feature = 4
   local kernel = kernel or 3
   local stride = stride or 1
   local pad = pad or 0
   local temporal = nn.TemporalConvolutionMM(
      input_feature, output_feature, kernel, stride, pad)
   print('Created module: '..tostring(temporal))
   temporal.gradWeight:zero()
   temporal.gradBias:zero()
   
   local output_length = 16
   local input_length = (output_length - 1) * stride + kernel - 2 * pad
   
   local input = torch.rand(batch, input_feature, input_length)
   print('Input size:')
   print(input:size())
   
   print('Executing forward propagation')
   local output = temporal:forward(input)
   print('Output size: ')
   print(output:size())

   local input_pad = torch.zeros(batch, input_feature, input_length + 2 * pad)
   input_pad:narrow(3, pad + 1, input_length):copy(input)
   local weight = temporal.weight:view(
      1, output_feature, input_feature, kernel):expand(
      batch, output_feature, input_feature, kernel)
   for i = 1, output:size(3) do
      local input_begin = (i - 1) * stride + 1
      local input_chunk = input_pad:narrow(
	 3, input_begin, kernel):contiguous():view(
	 batch, 1, input_feature, kernel):expand(
	 batch, output_feature, input_feature, kernel)
      local output_slice = torch.cmul(
	 weight, input_chunk):sum(4):sum(3):squeeze()
      output_slice:add(
	 1, temporal.bias:view(1, output_feature):expandAs(output_slice))
      print('Error of output slice '..i..': '..
	       output_slice:add(-1, output:select(3, i)):abs():mean())
   end

   local grad_output = torch.rand(output:size())
   print('Executing backward propagation')
   local grad_input = temporal:backward(input, grad_output)
   print('Input gradient size: ')
   print(grad_input:size())

   local grad_output_pad = torch.Tensor(
      batch, output_feature, input_length + kernel - 1):zero()
   local interlace_length = stride * (grad_output:size(3) - 1) + 1
   local interlace_shift = (grad_output_pad:size(3) - interlace_length) / 2
   for i = 1, grad_output:size(3) do
      local grad_output_pad_begin = (i - 1) * stride + 1 + interlace_shift
      if grad_output_pad_begin >= 1
	 and grad_output_pad_begin <= grad_output_pad:size(3) then
	 grad_output_pad:select(3, grad_output_pad_begin):copy(
	    grad_output:select(3, i))
      end
   end
   local weight_reverse = torch.Tensor(temporal.weight:size())
   local weight_index = torch.LongTensor(kernel)
   for i = 1, weight_index:size(1) do
      weight_index[i] = kernel - i + 1
   end
   weight_reverse:indexCopy(3, weight_index, temporal.weight)
   for i = 1, grad_input:size(3) do
      local grad_output_pad_begin = i
      local grad_output_pad_chunk = grad_output_pad:narrow(
	 3, grad_output_pad_begin, kernel):contiguous():view(
	 batch, output_feature, 1, kernel):expand(
	 batch, output_feature, input_feature, kernel)
      local grad_input_slice = torch.cmul(
	 weight_reverse:view(1, output_feature, input_feature, kernel):expand(
	    batch, output_feature, input_feature, kernel),
	 grad_output_pad_chunk):sum(4):sum(2):squeeze()
      print('Error of input gradient slice '..i..': '..
	       grad_input_slice:add(-1, grad_input:select(3, i)):abs():mean())
   end

   local input_unfold = input_pad:unfold(3, kernel, stride)
   for i = 1, temporal.weight:size(3) do
      local grad_weight_slice = torch.bmm(
	 grad_output, input_unfold:select(4, i):transpose(2, 3)):sum(
	 1):squeeze()
      print('Error of weight gradient slice '..i..': '..grad_weight_slice:add(
		  -1, temporal.gradWeight:select(3, i)):abs():mean())
   end
   
   local grad_bias = grad_output:sum(3):sum(1)
   print('Error of bias gradient: '..grad_bias:add(
	       -1, temporal.gradBias):abs():mean())

   local jacobian = self.jacobian
   local err = jacobian.testJacobian(temporal, input)
   print('Error of jacobian test: '..err)
   local err = jacobian.testJacobianParameters(
      temporal, input, temporal.weight, temporal.gradWeight)
   print('Error of jacobian test for weight: '..err)
   local err = jacobian.testJacobianParameters(
      temporal, input, temporal.bias, temporal.gradBias)
   print('Error of jacobian test for bias: '..err)
   local err = jacobian.testJacobianUpdateParameters(
      temporal, input, temporal.weight)
   print('Error of jacobian test for weight update: '..err)
   local err = jacobian.testJacobianUpdateParameters(
      temporal, input, temporal.bias)
   print('Error of jacobian test for bias update: '..err)
   for t,err in pairs(
      jacobian.testAllUpdate(temporal, input, 'weight', 'gradWeight')) do
      print('Error of jacobian test for '..t..' all update: '..err)
   end
   for t,err in pairs(
      jacobian.testAllUpdate(temporal, input, 'bias', 'gradBias')) do
      print('Error of jacobian test for '..t..' all update: '..err)
   end
end

function joe:batchCPUTest()
   for _, kernel in ipairs({3, 5}) do
      for _, stride in ipairs({1, 2, 3, 5}) do
	 for _, pad in ipairs({0, 1, 2, 3, 5}) do
	    self:batchCPU(kernel, stride, pad)
	 end
      end
   end
end

function joe:noBatchGPU(kernel, stride, pad)
   local input_feature = 2
   local output_feature = 4
   local kernel = kernel or 3
   local stride = stride or 1
   local pad = pad or 0
   local temporal = nn.TemporalConvolutionMM(
      input_feature, output_feature, kernel, stride, pad):cuda()
   print('Created module: '..tostring(temporal))
   temporal.gradWeight:zero()
   temporal.gradBias:zero()
   
   local output_length = 16
   local input_length = (output_length - 1) * stride + kernel - 2 * pad
   
   local input = torch.rand(input_feature, input_length):cuda()
   print('Input size:')
   print(input:size())
   
   print('Executing forward propagation')
   local output = temporal:forward(input)
   print('Output size: ')
   print(output:size())
   
   local input_pad = torch.zeros(input_feature, input_length + 2 * pad):cuda()
   input_pad:narrow(2, pad + 1, input_length):copy(input)
   for i = 1, output:size(2) do
      local input_begin = (i - 1) * stride + 1
      local input_chunk = input_pad:narrow(
         2, input_begin, kernel):contiguous():view(
         1, input_feature, kernel):expand(output_feature, input_feature, kernel)
      local output_slice = torch.cmul(
         temporal.weight, input_chunk):sum(3):sum(2):squeeze()
      output_slice:add(1, temporal.bias:viewAs(output_slice))
      print('Error of output slice '..i..': '..
               output_slice:add(-1, output:select(2, i)):abs():mean())
   end
   
   local grad_output = torch.rand(output:size()):cuda()
   print('Executing backward propagation')
   local grad_input = temporal:backward(input, grad_output)
   print('Input gradient size: ')
   print(grad_input:size())
   
   local grad_output_pad = torch.Tensor(
      output_feature, input_length + kernel - 1):zero():cuda()
   local interlace_length = stride * (grad_output:size(2) - 1) + 1
   local interlace_shift = (grad_output_pad:size(2) - interlace_length) / 2
   for i = 1, grad_output:size(2) do
      local grad_output_pad_begin = (i - 1) * stride + 1 + interlace_shift
      if grad_output_pad_begin >= 1
         and grad_output_pad_begin <= grad_output_pad:size(2) then
            grad_output_pad:select(2, grad_output_pad_begin):copy(
               grad_output:select(2, i))
      end
   end
   local weight_reverse = torch.Tensor(temporal.weight:size()):cuda()
   local weight_index = torch.LongTensor(kernel)
   for i = 1, weight_index:size(1) do
      weight_index[i] = kernel - i + 1
   end
   weight_reverse:indexCopy(3, weight_index, temporal.weight)
   for i = 1, grad_input:size(2) do
      local grad_output_pad_begin = i
      local grad_output_pad_chunk = grad_output_pad:narrow(
         2, grad_output_pad_begin, kernel):contiguous():view(
         output_feature, 1, kernel):expand(
         output_feature, input_feature, kernel)
      local grad_input_slice = torch.cmul(
         weight_reverse, grad_output_pad_chunk):sum(3):sum(1):squeeze()
      print('Error of input gradient slice '..i..': '..
               grad_input_slice:add(-1, grad_input:select(2, i)):abs():mean())
   end
   
   local input_unfold = input_pad:unfold(2, kernel, stride)
   for i = 1, temporal.weight:size(3) do
      local grad_weight_slice = torch.mm(
         grad_output, input_unfold:select(3, i):transpose(1, 2))
      print('Error of weight gradient slice '..i..': '..grad_weight_slice:add(
                  -1, temporal.gradWeight:select(3, i)):abs():mean())
   end
   
   local grad_bias = grad_output:sum(2)
   print('Error of bias gradient: '..grad_bias:add(
               -1, temporal.gradBias):abs():mean())
end

function joe:noBatchGPUTest()
   for _, kernel in ipairs({3, 5}) do
      for _, stride in ipairs({1, 2, 3, 5}) do
	 for _, pad in ipairs({0, 1, 2, 3, 5}) do
	    self:noBatchGPU(kernel, stride, pad)
	 end
      end
   end
end

function joe:batchGPU(kernel, stride, pad)
   local batch = 4
   local input_feature = 2
   local output_feature = 4
   local kernel = kernel or 3
   local stride = stride or 1
   local pad = pad or 0
   local temporal = nn.TemporalConvolutionMM(
      input_feature, output_feature, kernel, stride, pad):cuda()
   print('Created module: '..tostring(temporal))
   temporal.gradWeight:zero()
   temporal.gradBias:zero()
   
   local output_length = 16
   local input_length = (output_length - 1) * stride + kernel - 2 * pad
   
   local input = torch.rand(batch, input_feature, input_length):cuda()
   print('Input size:')
   print(input:size())
   
   print('Executing forward propagation')
   local output = temporal:forward(input)
   print('Output size: ')
   print(output:size())

   local input_pad = torch.zeros(
      batch, input_feature, input_length + 2 * pad):cuda()
   input_pad:narrow(3, pad + 1, input_length):copy(input)
   local weight = temporal.weight:view(
      1, output_feature, input_feature, kernel):expand(
      batch, output_feature, input_feature, kernel)
   for i = 1, output:size(3) do
      local input_begin = (i - 1) * stride + 1
      local input_chunk = input_pad:narrow(
         3, input_begin, kernel):contiguous():view(
         batch, 1, input_feature, kernel):expand(
         batch, output_feature, input_feature, kernel)
      local output_slice = torch.cmul(
         weight, input_chunk):sum(4):sum(3):squeeze()
      output_slice:add(
         1, temporal.bias:view(1, output_feature):expandAs(output_slice))
      print('Error of output slice '..i..': '..
               output_slice:add(-1, output:select(3, i)):abs():mean())
   end

   local grad_output = torch.rand(output:size()):cuda()
   print('Executing backward propagation')
   local grad_input = temporal:backward(input, grad_output)
   print('Input gradient size: ')
   print(grad_input:size())

   local grad_output_pad = torch.Tensor(
      batch, output_feature, input_length + kernel - 1):zero():cuda()
   local interlace_length = stride * (grad_output:size(3) - 1) + 1
   local interlace_shift = (grad_output_pad:size(3) - interlace_length) / 2
   for i = 1, grad_output:size(3) do
      local grad_output_pad_begin = (i - 1) * stride + 1 + interlace_shift
      if grad_output_pad_begin >= 1
         and grad_output_pad_begin <= grad_output_pad:size(3) then
            grad_output_pad:select(3, grad_output_pad_begin):copy(
               grad_output:select(3, i))
      end
   end
   local weight_reverse = torch.Tensor(temporal.weight:size()):cuda()
   local weight_index = torch.LongTensor(kernel)
   for i = 1, weight_index:size(1) do
      weight_index[i] = kernel - i + 1
   end
   weight_reverse:indexCopy(3, weight_index, temporal.weight)
   for i = 1, grad_input:size(3) do
      local grad_output_pad_begin = i
      local grad_output_pad_chunk = grad_output_pad:narrow(
         3, grad_output_pad_begin, kernel):contiguous():view(
         batch, output_feature, 1, kernel):expand(
         batch, output_feature, input_feature, kernel)
      local grad_input_slice = torch.cmul(
         weight_reverse:view(1, output_feature, input_feature, kernel):expand(
            batch, output_feature, input_feature, kernel),
         grad_output_pad_chunk):sum(4):sum(2):squeeze()
      print('Error of input gradient slice '..i..': '..
               grad_input_slice:add(-1, grad_input:select(3, i)):abs():mean())
   end

   local input_unfold = input_pad:unfold(3, kernel, stride)
   for i = 1, temporal.weight:size(3) do
      local grad_weight_slice = torch.bmm(
         grad_output, input_unfold:select(4, i):transpose(2, 3)):sum(
         1):squeeze()
      print('Error of weight gradient slice '..i..': '..grad_weight_slice:add(
                  -1, temporal.gradWeight:select(3, i)):abs():mean())
   end
   
   local grad_bias = grad_output:sum(3):sum(1)
   print('Error of bias gradient: '..grad_bias:add(
               -1, temporal.gradBias):abs():mean())
end

function joe:batchGPUTest()
   for _, kernel in ipairs({3, 5}) do
      for _, stride in ipairs({1, 2, 3, 5}) do
	 for _, pad in ipairs({0, 1, 2, 3, 5}) do
	    self:batchGPU(kernel, stride, pad)
	 end
      end
   end
end

joe.main()
return joe

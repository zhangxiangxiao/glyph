--[[
Convert string serialization to code
Copyright 2016 Xiang Zhang

Usage: th convert_string_code.lua [input] [output]
--]]

local torch = require('torch')

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input = arg[1] or '../data/dianping/train_string.t7b'
   local output = arg[2] or '../data/dianping/train_string_code.t7b'

   print('Reading from '..input)
   local input_data = torch.load(input)
   print('Converting to code format')
   local output_data = joe.convert(input_data)
   print('Saving to '..output)
   torch.save(output, output_data)
end

function joe.convert(input_data)
   local output_data = {}
   output_data.code = input_data.index
   output_data.code_value = input_data.content
   return output_data
end

joe.main()
return joe

--[[
Limit the maximum code value
Copyright 2016 Xiang Zhang

Usage: th limit_code.lua [input] [output] [limit]
--]]

local torch = require('torch')

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input = arg[1] or '../data/dianping/train_word.t7b'
   local output = arg[2] or '../data/dianping/train_word_limit.t7b'
   local limit = arg[3] and tonumber(arg[3]) or 200000

   print('Loading data from '..input)
   local data = torch.load(input)

   print('Limiting code to '..limit)
   local code = joe.limitCode(data, limit)

   print('Saving to '..output)
   torch.save(output, code)
end

function joe.limitCode(data, limit)
   local code, code_value = data.code, data.code_value
   local preserve = code_value:le(limit):long()
   local replace = code_value:gt(limit):long()
   code_value:cmul(preserve):add(replace:mul(limit + 1))
   return {code = code, code_value = code_value}
end

joe.main()
return joe

--[[
Construct unicode serialization format from string serialization format
Copyright 2015-2016 Xiang Zhang

Usage: th construct_code.lua [input] [output] [limit] [replace]
--]]

local bit32 = require('bit32')
local ffi = require('ffi')
local math = require('math')
local torch = require('torch')

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input = arg[1] or '../data/dianping/train_string.t7b'
   local output = arg[2] or '../data/dianping/train_code.t7b'
   local limit = arg[3] and tonumber(arg[3]) or 65536
   local replace = arg[4] and tonumber(arg[4]) or 33

   print('Loading data from '..input)
   local data = torch.load(input)

   print('Counting UTF-8 code')
   local count = joe.countCode(data)
   print('Total number of codes: '..count)

   print('Constructing UTF-8 code data')
   local code = joe.constructCode(data, count, limit, replace)

   print('Saving to '..output)
   torch.save(output, code)
end

function joe.countCode(data)
   local index, content = data.index, data.content

   local count = 0
   -- Iterate through the classes
   for i = 1, #index do
      print('Processing for class '..i)
      -- Iterate through the samples
      for j = 1, index[i]:size(1) do
         if math.fmod(j, 10000) == 0 then
            io.write('\rProcessing text: ', j, '/', index[i]:size(1))
            io.flush()
         end
         -- Iterate through the fields
         for k = 1, index[i][j]:size(1) do
            local text = ffi.string(
               torch.data(content:narrow(1, index[i][j][k][1], 1)))
            local sequence = joe.utf8to32(text)
            count = count + #sequence
         end
      end
      print('\rProcessed texts: '..index[i]:size(1)..'/'..index[i]:size(1))
   end

   return count
end

function joe.constructCode(data, count, limit, replace)
   local index, content = data.index, data.content
   local code = {}
   local code_value = torch.LongTensor(count)

   local p = 1
   -- Iterate through the classes
   for i = 1, #index do
      print('Processing for class '..i)
      code[i] = index[i]:clone():zero()
      -- Iterate through the samples
      for j = 1, index[i]:size(1) do
         if math.fmod(j, 10000) == 0 then
            io.write('\rProcessing text: ', j, '/', index[i]:size(1))
            io.flush()
         end
         -- Iterate through the fields
         for k = 1, index[i][j]:size(1) do
            local text = ffi.string(
               torch.data(content:narrow(1, index[i][j][k][1], 1)))
            local sequence = joe.utf8to32(text)
            code[i][j][k][1] = p
            code[i][j][k][2] = #sequence
            for l = 1, #sequence do
               code_value[p + l - 1] = sequence[l] + 1
               if limit and code_value[p + l - 1] > limit then
                  code_value[p + l - 1] = replace
               end
            end
            p = p + #sequence
         end
      end
      print('\rProcessed texts: '..index[i]:size(1)..'/'..index[i]:size(1))
   end

   return {code = code, code_value = code_value}
end

-- UTF-8 decoding function
-- Ref: http://lua-users.org/wiki/LuaUnicode
function joe.utf8to32(utf8str)
   assert(type(utf8str) == 'string')
   local res, seq, val = {}, 0, nil
   for i = 1, #utf8str do
      local c = string.byte(utf8str, i)
      if seq == 0 then
         table.insert(res, val)
         seq = c < 0x80 and 1 or c < 0xE0 and 2 or c < 0xF0 and 3 or
            c < 0xF8 and 4 or --c < 0xFC and 5 or c < 0xFE and 6 or
            error('Invalid UTF-8 character sequence')
         val = bit32.band(c, 2^(8-seq) - 1)
      else
         val = bit32.bor(bit32.lshift(val, 6), bit32.band(c, 0x3F))
      end
      seq = seq - 1
   end
   table.insert(res, val)
   table.insert(res, 0)
   return res
end

joe.main()
return joe

--[[
Visualizing argument string using GNU Unifont
Copyright 2015 Xiang Zhang
--]]

local bit32 = require('bit32')
local image = require('image')
local torch = require('torch')

local joe = {}

function joe.main()
   local input = arg[1]
   local unifont = arg[2] or 'unifont/unifont-8.0.01.t7b'

   print('Loading unifont from '..unifont)
   local data = torch.load(unifont)
   local sequence = joe.utf8to32(input)
   local im = torch.Tensor(data:size(2), data:size(3) * #sequence)
   for i, c in ipairs(sequence) do
      im:narrow(2, 1 + (i-1)*data:size(3), data:size(3)):copy(data[c + 1])
   end
   print('Visualizing')
   image.display({image = im, zoom = 4})
end

-- Ref: http://lua-users.org/wiki/LuaUnicode
function joe.utf8to32(utf8str)
   assert(type(utf8str) == "string")
   local res, seq, val = {}, 0, nil
   for i = 1, #utf8str do
      local c = string.byte(utf8str, i)
      if seq == 0 then
	 table.insert(res, val)
	 seq = c < 0x80 and 1 or c < 0xE0 and 2 or c < 0xF0 and 3 or
	    c < 0xF8 and 4 or --c < 0xFC and 5 or c < 0xFE and 6 or
	    error("invalid UTF-8 character sequence")
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
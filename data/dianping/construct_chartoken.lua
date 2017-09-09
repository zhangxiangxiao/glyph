--[[
Create chartoken format for fastText
Copyright 2017 Xiang Zhang

Usage: th construct_chartoken.lua [input] [output]
--]]

local bit32 = require('bit32')
local io = require('io')
local math = require('math')
local string = require('string')
local torch = require('torch')

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input = arg[1] or '../data/dianping/train.csv'
   local output = arg[2] or '../data/dianping/train_chartoken.txt'

   print('Construct token')
   joe.constructToken(input, output)
end

function joe.constructToken(input, output)
   local ifd = io.open(input)
   local ofd = io.open(output, 'w')

   local n = 0
   for line in ifd:lines() do
      n = n + 1
      if math.fmod(n, 10000) == 0 then
         io.write('\rProcessing line: ', n)
         io.flush()
      end

      local content = joe.parseCSVLine(line)
      local class = tonumber(content[1])

      ofd:write('__label__', class)
      for i = 2, #content do
         content[i] = content[i]:gsub('\\n', ' '):gsub(
            '[%z\001-\031\127]', ' '):gsub('^%s*(.-)%s*$', '%1')
         local sequence = joe.utf8to32(content[i])
         for j, code in ipairs(sequence) do
            if code > 32 then
               ofd:write(' ', joe.utf8str(code))
            end
         end
      end

      ofd:write('\n')
   end
   print('\rProcessed lines: '..n)
   ifd:close()
   ofd:close()
end

-- Parsing csv line
-- Ref: http://lua-users.org/wiki/LuaCsv
function joe.parseCSVLine(line,sep) 
   local res = {}
   local pos = 1
   sep = sep or ','
   while true do 
      local c = string.sub(line,pos,pos)
      if (c == "") then break end
      if (c == '"') then
         -- quoted value (ignore separator within)
         local txt = ""
         repeat
            local startp,endp = string.find(line,'^%b""',pos)
            txt = txt..string.sub(line,startp+1,endp-1)
            pos = endp + 1
            c = string.sub(line,pos,pos) 
            if (c == '"') then txt = txt..'"' end 
            -- check first char AFTER quoted string, if it is another
            -- quoted string without separator, then append it
            -- this is the way to "escape" the quote char in a quote.
         until (c ~= '"')
         table.insert(res,txt)
         assert(c == sep or c == "")
         pos = pos + 1
      else
         -- no quotes used, just look for the first separator
         local startp,endp = string.find(line,sep,pos)
         if (startp) then 
            table.insert(res,string.sub(line,pos,startp-1))
            pos = endp + 1
         else
            -- no separator found -> use rest of string and terminate
            table.insert(res,string.sub(line,pos))
            break
         end 
      end
   end
   return res
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

-- UTF-8 encoding function
-- Ref: http://stackoverflow.com/questions/7983574/how-to-write-a-unicode-symbol
--      -in-lua
function joe.utf8str(decimal)
   local bytemarkers = {{0x7FF, 192}, {0xFFFF, 224}, {0x1FFFFF, 240}}
   if decimal < 128 then return string.char(decimal) end
   local charbytes = {}
   for bytes,vals in ipairs(bytemarkers) do
      if decimal <= vals[1] then
         for b = bytes + 1, 2, -1 do
            local mod = decimal % 64
            decimal = (decimal - mod) / 64
            charbytes[b] = string.char(128+mod)
         end
         charbytes[1] = string.char(vals[2] + decimal)
         break
      end
   end
   return table.concat(charbytes)
end

joe.main()
return joe

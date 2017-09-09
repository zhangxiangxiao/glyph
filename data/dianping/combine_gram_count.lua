--[[
Combine sorted gram counts
Copyright 2016 Xiang Zhang

Usage: th combine_gram_count.lua [input_prefix] [output] [samples] [chunks]

Comment: This program also outputs lines with counts as the firt unquoted csv
   value, so that one can use GNU sort easily.
--]]

local io = require('io')
local math = require('math')
local string = require('string')

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input_prefix = arg[1] or '../data/dianping/train_chargram_count_sort/'
   local output = arg[2] or '../data/dianping/train_chargram_count_combine.csv'
   local samples = arg[3] and tonumber(arg[3]) or 2000000
   local chunks = arg[4] and tonumber(arg[4]) or 100

   print('Combine chunks')
   joe.combineChunks(input_prefix, output, samples, chunks)
end

function joe.combineChunks(input_prefix, output, samples, chunks)
   local n = 0
   local ofd = io.open(output, 'w')
   local current = {}
   for i = 1, chunks do
      local ifd = io.open(input_prefix..i..'.csv')
      for line in ifd:lines() do
         n = n + 1
         if math.fmod(n, 100000) == 0 then
            io.write('\rProcessing line ', n)
            io.flush()
         end
         local content = joe.parseCSVLine(line)
         if current[1] ~= content[1] then
            if current[1] ~= nil then
               ofd:write(current[3], ',"', current[1], '","',
                         current[2]:gsub('"', '""'), '","',
                         current[4] / samples, '","', current[3], '"\n')
            end
            current = content
         else
            current[3] = current[3] + content[3]
            current[4] = current[4] + content[4]
         end
      end
      ifd:close()
   end
   ofd:write(current[3], ',"', current[1], '","',
             current[2]:gsub('"', '""'), '","',
             current[4] / samples, '","', current[3], '"\n')
   ofd:close()
   print('\rProcessed lines: '..n)
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

joe.main()
return joe

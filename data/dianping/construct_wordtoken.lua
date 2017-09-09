--[[
Construct word token format from csv files
Copyright 2017 Xiang Zhang

Usage: th construct_wordtoken [input] [list] [output]
--]]

local io = require('io')
local math = require('math')
local tds = require('tds')
local torch = require('torch')

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input = arg[1] or '../data/dianping/train_word.csv'
   local list = arg[2] or '../data/dianping/train_word_list.csv'
   local output = arg[3] or '../data/dianping/train_wordtoken.txt'

   print('Reading list from '..list)
   local word_list = joe.readList(list)

   print('Constructing word token')
   joe.constructToken(input, output, word_list)
end

function joe.readList(list)
   local word_list = tds.Vec()
   local fd = io.open(list)
   local n = 0
   for line in fd:lines() do
      n = n + 1
      if math.fmod(n, 10000) == 0 then
         io.write('\rProcessing line: ', n)
         io.flush()
      end

      local content = joe.parseCSVLine(line)
      word_list[#word_list + 1] =
         content[1]:gsub('\\n', '\n'):gsub('[%z\001-\032\127]', ' '):gsub(
            '^%s*(.-)%s*$', '%1')
   end
   print('\rProcessed lines: '..n)
   fd:close()

   return word_list
end

function joe.constructToken(input, output, word_list)
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
         content[i] = content[i]:gsub('\\n', '\n'):gsub('^%s*(.-)%s*$', '%1')
         for word in content[i]:gmatch('%d+') do
            local word_string = word_list[tonumber(word)] or '<unk>'
            ofd:write(' ', word_string)
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

joe.main()
return joe


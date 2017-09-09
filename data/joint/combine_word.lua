--[[
Combine two word data together
Copyright 2016 Xiang Zhang

Usage: th combine_word_list.lua [input_1] [list_1] [input_2] [list_2] ...
   [output] [list]
--]]

local io = require('io')
local math = require('math')
local tds = require('tds')
local torch = require('torch')

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input = {}
   local input_list = {}
   for i = 1, math.floor(#arg / 2) - 1 do
      input[i] = arg[2 * i - 1]
      input_list[i] = arg[2 * i]
   end
   local output = arg[math.floor(#arg / 2) * 2 - 1] or
      '../data/joint/binary_train_word.csv'
   local output_list = arg[math.floor(#arg / 2) * 2] or
      '../data/joint/binary_train_word_list.csv'

   print('Loading output list from '..output_list)
   local list, count, freq, dict = joe.readList(output_list)
   print('Opening output file '..output)
   local ofd = io.open(output, 'w')

   for i = 1, #input do
      print('Loading input list from '..input_list[i])
      local local_list, local_count, local_freq, local_dict =
         joe.readList(input_list[i])
      print('Building input to output map')
      local map = joe.buildMap(local_list, dict)
      print('Processing data from '..input[i])
      joe.processInput(input[i], map, ofd, list)
   end

   print('Closing output file '..output)
   ofd:close()
end

function joe.readList(file)
   local list = tds.Vec()
   local count = tds.Vec()
   local freq = tds.Vec()
   local dict = tds.Hash()
   local fd = io.open(file)
   for line in fd:lines() do
      local content = joe.parseCSVLine(line)
      content[1] = content[1]:gsub('\\n', '\n')
      list:insert(content[1])
      count:insert(tonumber(content[2]))
      freq:insert(tonumber(content[3]))
      dict[content[1]] = #list
   end
   fd:close()
   return list, count, freq, dict
end

function joe.buildMap(input_list, dict)
   local map = tds.Vec()
   for i = 1, #input_list do
      map[i] = dict[input_list[i]]
   end
   return map
end

function joe.processInput(input, map, ofd, list)
   local ifd = io.open(input)
   local n = 0
   for line in ifd:lines() do
      n = n + 1
      if math.fmod(n, 10000) == 0 then
         io.write('\rProcessing line: ', n)
         io.flush()
      end

      -- Write class
      local content = joe.parseCSVLine(line)
      ofd:write('"', content[1], '"')

      -- Write title and comment
      for i = 2, #content do
         ofd:write(',"')
         for word in content[i]:gmatch('%d+') do
            ofd:write(map[tonumber(word)] or #list + 1, ' ')
         end
         ofd:write('"')
      end

      -- Write end of line
      ofd:write('\n')
   end
   print('\rProcessed lines: '..n)
   ifd:close()
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

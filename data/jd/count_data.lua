--[[
Count data for each class and length
Copyright 2016 Xiang Zhang

Usage: th count_data.lua [input] [output]
--]]

local torch = require('torch')

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input = arg[1] or '../data/jd/sentiment/comment_sorted_nonull.csv'
   local output = arg[2] or '../data/jd/sentiment/comment_sorted_count.t7b'

   print('Counting data')
   local count = joe.count(input)
   joe.count = count
   print('Saving to '..output)
   torch.save(output, count)
   print('Plotting result')
   joe.plot(count)
end

function joe.count(input)
   local count = {}
   local max_class = 0
   local max_length = 0
   local fd = io.open(input)
   local n = 0
   for line in fd:lines() do
      n = n + 1
      if math.fmod(n, 100000) == 0 then
         io.write('\rProcessing line: ', n)
         io.flush()
      end

      local content = joe.parseCSVLine(line)
      local class = tonumber(content[1])
      local length = 0
      for i = 2, #content do
         length = length + content[i]:gsub("^%s*(.-)%s*$", "%1"):len()
      end
      count[class] = count[class] or {}
      count[class][length] = (count[class][length] or 0) + 1

      if class > max_class then
         max_class = class
      end
      if length > max_length then
         max_length = length
      end
   end
   print('\rProcessed lines: '..n)
   print('total classes = '..max_class..', maximum length = '..max_length)
   fd:close()

   local result = torch.Tensor(max_class, max_length):zero()
   for class, class_count in pairs(count) do
      if class > 0 then
         for length, length_count in pairs(class_count) do
            if length > 0 then
               result[class][length] = length_count
            end
         end
      end
   end

   return result
end

function joe.plot(count)
   require('gnuplot')
   local cumulated = count:cumsum(2)
   local plots = {}
   for class = 1, cumulated:size(1) do
      plots[class] = {tostring(class), cumulated[class], '-'}
   end
   local figure = gnuplot.figure()
   gnuplot.plot(unpack(plots))
end

-- Parsing csv line
-- Ref: http://lua-users.org/wiki/LuaCsv
function joe.parseCSVLine (line,sep)
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

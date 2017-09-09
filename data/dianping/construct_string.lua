--[[
Create string serialization format from csv files
Copyright 2015-2016 Xiang Zhang

Usage: th construct_string.lua [input] [output]
--]]

local ffi = require('ffi')
local io = require('io')
local math = require('math')
local torch = require('torch')

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input = arg[1] or '../data/dianping/train.csv'
   local output = arg[2] or '../data/dianping/train_string.t7b'

   print('Counting samples')
   local count, bytes, fields = joe.countSamples(input)
   for i, v in ipairs(count) do
      print('Number of samples in class '..i..': '..v)
   end
   print('Total number of bytes: '..bytes)
   print('Number of text fields: '..fields)

   print('Constructing data')
   local data = joe.constructData(input, count, bytes, fields)
   print('Saving to '..output)
   torch.save(output, data)
end

function joe.countSamples(input)
   local count = {}
   local bytes = 0
   local fields = nil
   local n = 0
   local fd = io.open(input)
   for line in fd:lines() do
      n = n + 1
      if math.fmod(n, 10000) == 0 then
         io.write('\rProcessing line: ', n)
         io.flush()
      end

      local content = joe.parseCSVLine(line)
      local class = tonumber(content[1])

      count[class] = count[class] and count[class] + 1 or 1
      for i = 2, #content do
         content[i] = content[i]:gsub('\\n', '\n'):gsub("^%s*(.-)%s*$", "%1")
         bytes = bytes + content[i]:len() + 1
      end
      fields = fields or #content - 1
      if fields ~= #content - 1 then
         error('Number of fields is not '..fields..' at line '..n)
      end
   end
   print('\rProcessed lines: '..n)
   fd:close()

   return count, bytes, fields
end

function joe.constructData(input, count, bytes, fields)
   local data = torch.ByteTensor(bytes)
   local index = {}
   for i, v in ipairs(count) do
      index[i] = torch.LongTensor(v, fields, 2)
   end

   local progress = {}
   local n = 0
   local p = 1
   local fd = io.open(input)
   for line in fd:lines() do
      n = n + 1
      if math.fmod(n, 10000) == 0 then
         io.write('\rProcessing line: ', n)
         io.flush()
      end

      local content = joe.parseCSVLine(line)
      local class = tonumber(content[1])

      progress[class] = progress[class] and progress[class] + 1 or 1
      for i = 2, #content do
         content[i] = content[i]:gsub('\\n', '\n'):gsub("^%s*(.-)%s*$", "%1")
         index[class][progress[class]][i - 1][1] = p
         index[class][progress[class]][i - 1][2] = content[i]:len()
         ffi.copy(torch.data(data:narrow(1, p, content[i]:len() + 1)),
                  content[i])
         p = p + content[i]:len() + 1
      end
   end
   print('\rProcessed lines: '..n)
   fd:close()

   return {content = data, index = index}
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

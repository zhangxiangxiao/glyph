--[[
Split data into training and testing subsets
Copyright 2015 Xiang Zhang

Usage: th split_train [count] [input] [train] [test]
--]]

local io = require('io')
local math = require('math')
local torch = require('torch')

-- A Logic Named Joe
local joe = {}

function joe.main()
   local count = arg[1] or '../data/dianping/data_count.csv'
   local input = arg[2] or '../data/dianping/data.csv'
   local train = arg[3] or '../data/dianping/train.csv'
   local test = arg[4] or '../data/dianping/test.csv'

   local index = {}
   local cfd = io.open(count)
   for line in cfd:lines() do
      local content = joe.parseCSVLine(line)
      local class = tonumber(content[1])
      local total = tonumber(content[2])
      local train_count = tonumber(content[3])
      local test_count = tonumber(content[4])

      print('Constructing index '..class..': '..train_count..
               ','..test_count..','..total)
      index[class] = torch.ByteTensor(total):zero()
      local perm = torch.randperm(total)
      for i = 1, test_count do
         index[class][perm[i]] = 1
      end
   end
   cfd:close()

   local n = 0
   local progress = {}
   local ifd = io.open(input)
   local trfd = io.open(train, 'w')
   local tefd = io.open(test, 'w')
   for line in ifd:lines() do
      n = n + 1
      if math.fmod(n, 100000) == 0 then
         io.write('\rProcessing line: ', n)
         io.flush()
      end

      local content = joe.parseCSVLine(line)
      local class = tonumber(content[1])

      progress[class] = progress[class] and progress[class] + 1 or 1
      if index[class] and index[class][progress[class]] == 0 then
         trfd:write(line, '\n')
      else
         tefd:write(line, '\n')
      end
   end
   print('\rProcessed lines: '..n)
   ifd:close()
   trfd:close()
   tefd:close()
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

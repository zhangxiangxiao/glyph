--[[
Construct tfidf format from bag format
Copyright 2016 Xiang Zhang

Usage: th construct_tfidf.lua [input] [output] [list] [limit]
--]]

local io = require('io')
local math = require('math')
local torch = require('torch')

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input = arg[1] or '../data/dianping/train_charbag.t7b'
   local output = arg[2] or '../data/dianping/train_charbagtfidf.t7b'
   local list = arg[3] or '../data/dianping/train_charbag_list.csv'
   local limit = arg[4] and tonumber(arg[4]) or 200000

   print('Loading data from '..input)
   local data = torch.load(input)

   print('Loading frequency list from '..list)
   local freq = joe.readList(list)
   print('Frequency list length '..freq:size(1))

   print('Constructing bag-of-elements TFIDF data')
   local tfidf = joe.constructTfidf(data, freq, limit)

   print('Saving to '..output)
   torch.save(output, tfidf)
end

function joe.readList(list)
   local freq = {}
   local fd = io.open(list)
   for line in fd:lines() do
      local content = joe.parseCSVLine(line)
      content[2] = content[2]:gsub('\\n', '\n')
      freq[#freq + 1] = tonumber(content[3])
   end
   return torch.Tensor(freq)
end

function joe.constructTfidf(data, freq, limit)
   local bag, bag_index, bag_value = data.bag, data.bag_index, data.bag_value
   local tfidf_value = bag_value:clone()

   local freq = freq
   if freq:size(1) > limit then
      freq:narrow(1, limit + 1, freq:size(1) - limit):zero()
   elseif freq:size(1) < limit + 1 then
      local new_freq = freq.new(limit + 1):zero()
      new_freq:narrow(1, 1, freq:size(1)):copy(freq)
      freq = new_freq
   end

   freq:apply(function (x) return x > 0 and math.log(1/x) or 0 end)
   local indexed = freq:index(1, bag_index)
   tfidf_value:cmul(indexed)

   -- Iterate through the classes
   for i = 1, #bag do
      print('Processing for class '..i)
      -- Iterate through the samples
      for j = 1, bag[i]:size(1) do
         if math.fmod(j, 10000) == 0 then
            io.write('\rProcessing sample: ', j, '/', bag[i]:size(1))
            io.flush()
         end
         if bag[i][j][2] > 0 and
         tfidf_value:narrow(1, bag[i][j][1], bag[i][j][2]):sum() ~= 0 then
            tfidf_value:narrow(1, bag[i][j][1], bag[i][j][2]):div(
               tfidf_value:narrow(1, bag[i][j][1], bag[i][j][2]):sum())
         end
      end
      print('\rProcessed samples: '..bag[i]:size(1)..'/'..bag[i]:size(1))
   end

   return {bag = bag, bag_index = bag_index, bag_value = tfidf_value}
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

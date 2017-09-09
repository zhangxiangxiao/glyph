--[[
Construct unicode character bag-of-element format from unicode serialization
Copyright 2016 Xiang Zhang

Usage: th construct_charbag.lua [input] [output] [list] [read] [limit] [replace]
--]]

local io = require('io')
local math = require('math')
local torch = require('torch')

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input = arg[1] or '../data/dianping/train_code.t7b'
   local output = arg[2] or '../data/dianping/train_charbag.t7b'
   local list = arg[3] or '../data/dianping/train_charbag_list.csv'
   local read = (arg[4] == 'true')
   local limit = arg[5] and tonumber(arg[5]) or 200000
   local replace = arg[6] and tonumber(arg[6]) or 200001

   print('Loading data from '..input)
   local data = torch.load(input)

   print('Counting character')
   local count, freq = joe.countBag(data, limit, replace)
   print('Total number of values: '..count)

   if read == true then
      print('Reading frequency from '..list)
      freq = joe.readList(list)
   else
      print('Outputing frequency list to '..list)
      joe.writeList(freq, list)
   end

   print('Constructing character bag data')
   local bag = joe.constructBag(data, count, limit, replace)

   print('Saving to '..output)
   torch.save(output, bag)
end

function joe.writeList(freq, list)
   local fd = io.open(list, 'w')
   for i = 1, freq:size(1) do
      local char = (i <= 65536) and joe.utf8str(i - 1) or ''
      -- Do not print control characters
      if i < 11 or (i > 11 and i < 33) then
         char = ''
      end
      fd:write('"', i, '","', char:gsub('\n', '\\n'):gsub('"', '""'), '","',
               freq[i], '"\n')
   end
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

function joe.countBag(data, limit, replace)
   local code, code_value = data.code, data.code_value

   local count = 0
   local freq = torch.zeros(math.max(limit, replace))
   -- Iterate through the classes
   for i = 1, #code do
      print('Processing for class '..i)
      -- Iterate through the samples
      for j = 1, code[i]:size(1) do
         if math.fmod(j, 1000) == 0 then
            io.write('\rProcessing text: ', j, '/', code[i]:size(1))
            io.flush()
         end
         local index = {}
         -- Iterate through the fields
         for k = 1, code[i][j]:size(1) do
            for l = 1, code[i][j][k][2] do
               local char = code_value[code[i][j][k][1] + l - 1]
               if char > limit then
                  char = replace
               end
               if not index[char] then
                  count = count + 1
                  index[char] = 1
                  freq[char] = freq[char] + 1
               else
                  index[char] = index[char] + 1
               end
            end
         end
      end
      print('\rProcessed texts: '..code[i]:size(1)..'/'..code[i]:size(1))
   end

   -- Normalizing the frequency
   local sum = 0
   for i = 1, #code do
      sum = sum + code[i]:size(1)
   end
   freq:div(sum)
   return count, freq
end

function joe.constructBag(data, count, limit, replace)
   local code, code_value = data.code, data.code_value
   local bag = {}
   local bag_index = torch.LongTensor(count)
   local bag_value = torch.DoubleTensor(count)

   local count = 0
   -- Iterate through the classes
   for i = 1, #code do
      print('Processing for class '..i)
      bag[i] = torch.LongTensor(code[i]:size(1), 2)
      -- Iterate through the samples
      for j = 1, code[i]:size(1) do
         if math.fmod(j, 1000) == 0 then
            io.write('\rProcessing text: ', j, '/', code[i]:size(1))
            io.flush()
         end
         local index = {}
         local pointer = {}
         bag[i][j][1] = count + 1
         -- Iterate through the fields
         for k = 1, code[i][j]:size(1) do
            for l = 1, code[i][j][k][2] do
               local char = code_value[code[i][j][k][1] + l - 1]
               if char > limit then
                  char = replace
               end
               if not index[char] then
                  count = count + 1
                  index[char] = 1
                  pointer[#pointer + 1] = char
               else
                  index[char] = index[char] + 1
               end
            end
         end
         table.sort(pointer)
         bag[i][j][2] = #pointer
         for m = 1, #pointer do
            bag_index[bag[i][j][1] + m - 1] = pointer[m]
            if pointer[m] > limit then
               bag_value[bag[i][j][1] + m - 1] = 0
            else
               bag_value[bag[i][j][1] + m - 1] = index[pointer[m]]
            end
         end
        if #pointer > 0 and
        bag_value:narrow(1, bag[i][j][1], bag[i][j][2]):sum() ~= 0 then
           bag_value:narrow(1, bag[i][j][1], bag[i][j][2]):div(
              bag_value:narrow(1, bag[i][j][1], bag[i][j][2]):sum())
        end
      end
      print('\rProcessed texts: '..code[i]:size(1)..'/'..code[i]:size(1))
   end

   return {bag = bag, bag_index = bag_index, bag_value = bag_value}
end

joe.bytemarkers = {{0x7FF, 192}, {0xFFFF, 224}, {0x1FFFFF, 240}}
function joe.utf8str(decimal)
   local bytemarkers = joe.bytemarkers
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

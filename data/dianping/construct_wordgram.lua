--[[
Constructngrams format from serialization
Copyright 2016 Xiang Zhang

Usage: th construct_wordgram.lua [input] [output] [list] [gram] [limit]
   [replace]
--]]

local io = require('io')
local math = require('math')
local tds = require('tds')
local torch = require('torch')

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input = arg[1] or '../data/dianping/train_word.t7b'
   local output = arg[2] or '../data/dianping/train_wordgram.t7b'
   local list = arg[3] or '../data/dianping/train_wordgram_list.csv'
   local gram = arg[4] and tonumber(arg[4]) or 5
   local limit = arg[5] and tonumber(arg[5]) or 1000000
   local replace = arg[6] and tonumber(arg[6]) or 1000001

   print('Loading data from '..input)
   local data = torch.load(input)

   print('Reading frequency from '..list)
   local freq, dict = joe.readList(list)

   print('Counting character ngrams data')
   local count = joe.countBag(data, dict, gram, limit, replace)
   print('Total number of ngrams in data is '..count)

   print('Constructing character bag data')
   local bag = joe.constructBag(data, dict, count, gram, limit, replace)

   print('Saving to '..output)
   torch.save(output, bag)
end

function joe.readList(list)
   local freq_table = tds.Vec()
   local dict = tds.Hash()
   local fd = io.open(list)
   for line in fd:lines() do
      local content = joe.parseCSVLine(line)
      content[2] = content[2]:gsub('\\n', '\n')
      freq_table[#freq_table + 1] = tonumber(content[3])
      dict[content[1]] = #freq_table
   end

   local freq = torch.Tensor(#freq_table)
   for i, v in ipairs(freq_table) do
      freq[i] = v
   end
   return freq, dict
end

function joe.countBag(data, dict, gram, limit, replace)
   local count = 0
   local code, code_value = data.code, data.code_value

   -- Iterate through the classes
   for i = 1, #code do
      print('Processing for class '..i)
      -- Iterate through the samples
      for j = 1, code[i]:size(1) do
         if math.fmod(j, 1000) == 0 then
            io.write('\rProcessing text: ', j, '/', code[i]:size(1))
            io.flush()
            collectgarbage()
         end
         local index = {}
         -- Iterate through the fields
         for k = 1, code[i][j]:size(1) do
            -- Iterate through the grams
            for n = 1, gram do
               -- Iterate through the positions
               for l = 1, code[i][j][k][2] - n + 1 do
                  local ngram = tostring(code_value[code[i][j][k][1] + l - 1])
                  for m = 2, n do
                     ngram = ngram..' '..tostring(
                        code_value[code[i][j][k][1] + l - 1 + m - 1])
                  end
                  local ngram_index = dict[ngram]
                  if ngram_index == nil or ngram_index > limit then
                     ngram_index = replace
                  end
                  if not index[ngram_index] then
                     index[ngram_index] = 0
                     count = count + 1
                  end
                  index[ngram_index] = index[ngram_index] + 1
               end
            end
         end
      end
      print('\rProcessed texts: '..code[i]:size(1)..'/'..code[i]:size(1))
   end

   return count
end

function joe.constructBag(data, dict, count, gram, limit, replace)
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
            collectgarbage()
         end
         local index = {}
         local pointer = {}
         bag[i][j][1] = count + 1
         -- Iterate through the fields
         for k = 1, code[i][j]:size(1) do
            -- Iterate through the grams
            for n = 1, gram do
               -- Iterate through the positions
               for l = 1, code[i][j][k][2] - n + 1 do
                  local ngram = tostring(code_value[code[i][j][k][1] + l - 1])
                  for m = 2, n do
                     ngram = ngram..' '..tostring(
                        code_value[code[i][j][k][1] + l - 1 + m - 1])
                  end
                  local ngram_index = dict[ngram]
                  if ngram_index == nil or ngram_index > limit then
                     ngram_index = replace
                  end
                  if not index[ngram_index] then
                     count = count + 1
                     index[ngram_index] = 0
                     pointer[#pointer + 1] = ngram_index
                  end
                  index[ngram_index] = index[ngram_index] + 1
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

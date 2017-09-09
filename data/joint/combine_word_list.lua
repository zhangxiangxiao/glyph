--[[
Combine two word data together
Copyright 2016 Xiang Zhang

Usage: th combine_word_list.lua [list_1] [size_1] [list_2] [size_2] ... [output]
--]]

local io = require('io')
local math = require('math')
local tds = require('tds')
local torch = require('torch')

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input_list = {}
   local input_size = {}
   for i = 1, math.floor(#arg / 2) do
      input_list[i] = arg[2 * i - 1]
      input_size[i] = arg[2 * i]
   end
   local output_list = arg[math.floor(#arg / 2) * 2 + 1] or
      '../data/joint/binary_train_word_list.csv'

   local word = {}
   for i = 1, #input_list do
      print('Loading list from '..input_list[i])
      local list, count, freq, dict = joe.readInputList(input_list[i])
      word[i] = {list = list, count = count, freq = freq, dict = dict}
   end
   print('Merging word lists')
   local list, count_table, freq_table, dict =
      joe.mergeWords(word, input_size)
   print('Writing merged word list to '..output_list)
   joe.writeOutputList(output_list, list, count_table, freq_table, dict)
end

function joe.readInputList(file)
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

function joe.writeOutputList(file, list, count_table, freq_table, dict)
   local fd = io.open(file, 'w')
   for index, word in ipairs(list) do
      fd:write('"', word:gsub('\n', '\\n'):gsub('"', '""'), '","',
               count_table[word], '","', freq_table[word], '"\n')
   end
   fd:close()
end

function joe.mergeWords(word, size)
   local total_size = 0
   for i, s in ipairs(size) do
      total_size = total_size + s
   end

   local list = tds.Vec()
   local count_table = tds.Hash()
   local freq_table = tds.Hash()
   for i, w in ipairs(word) do
      for j, v in ipairs(w.list) do
         if count_table[v] == nil then
            list:insert(v)
            count_table[v] = w.count[j]
            freq_table[v] = w.freq[j] * size[i] / total_size
         else
            count_table[v] = count_table[v] + w.count[j]
            freq_table[v] = freq_table[v] + w.freq[j] * size[i] / total_size
         end
         if math.fmod(j, 100000) == 0 then
            io.write('\rProcessing list ', i, ': ', j, '/', #w.list)
            io.flush()
         end
      end
      print('\rProcessed list '..i..': '..(#w.list)..'/'..(#w.list))
   end

   print('Sorting merged word list')
   list:sort(function(a, b) return count_table[a] > count_table[b] end)

   print('Constructing merged word dictionary')
   local dict = tds.Hash()
   for i, w in ipairs(list) do
      dict[w] = i
   end

   return list, count_table, freq_table, dict
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

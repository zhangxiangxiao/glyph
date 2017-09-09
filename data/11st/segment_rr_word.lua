--[[
Create romananized word data from romanized data in csv for Korean
Copyright 2016 Xiang Zhang

Usage: th segment_rr_word.lua [input] [output] [list] [read]
--]]

local ffi = require('ffi')
local io = require('io')
local math = require('math')
local tds = require('tds')
local torch = require('torch')

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input = arg[1] or '../data/11st/sentiment/full_train_rr.csv'
   local output = arg[2] or '../data/11st/sentiment/full_train_rr_word.csv'
   local list = arg[3] or '../data/11st/sentiment/full_train_rr_word_list.csv'
   local read = (arg[4] == 'true')

   local word_index, word_total
   if read then
      print('Reading word index')
      word_index, word_total = joe.readWords(list)
   else
      print('Counting words')
      local word_count, word_freq = joe.splitWords(input)
      print('Sorting words by count')
      word_index, word_total = joe.sortWords(list, word_count, word_freq)
   end

   print('Constructing word index output')
   joe.constructWords(input, output, word_index, word_total)
end

function joe.readWords(list)
   local word_index = tds.Hash()
   local fd = io.open(list)
   local n = 0
   for line in fd:lines() do
      n = n + 1
      if math.fmod(n, 10000) == 0 then
         io.write('\rProcessing line: '..n)
         io.flush()
      end

      local content = joe.parseCSVLine(line)
      content[1] = content[1]:gsub('\\n', '\n')
      word_index[content[1]] = n
   end
   print('\rProcessed lines: '..n)
   fd:close()
   return word_index, n
end

function joe.splitWords(input)
   local word_count, word_freq = tds.Hash(), tds.Hash()
   local fd = io.open(input)
   local n = 0
   for line in fd:lines() do
      n = n + 1
      if math.fmod(n, 10000) == 0 then
         io.write('\rProcessing line: ', n)
         io.flush()
      end

      local content = joe.parseCSVLine(line)
      field_set = {}
      for i = 2, #content do
         content[i] = content[i]:gsub('\\n', '\n'):gsub("^%s*(.-)%s*$", "%1")
         -- All punctuation characters except for hyphen "-"
         content[i] = content[i]:gsub(
            '([!"#$%%&\'()*+,./:;<=>?@%[\\%]^_`{|}~])', ' %1 ')
         for word in content[i]:gmatch('[%S]+') do
            word_count[word] = (word_count[word] or 0) + 1
            if not field_set[word] then
               field_set[word] = true
               word_freq[word] = (word_freq[word] or 0) + 1
            end
         end
      end
   end
   print('\rProcessed lines: '..n)
   fd:close()

   -- Normalizing word frequencies
   for key, value in pairs(word_freq) do
      word_freq[key] = value / n
   end

   return word_count, word_freq
end

function joe.sortWords(list, word_count, word_freq)
   -- Sort the list of words
   word_list = tds.Vec()
   for word, _ in pairs(word_count) do
      word_list[#word_list + 1] = word
   end
   word_list:sort(function (w, v) return word_count[w] > word_count[v] end)

   -- Create the word index
   word_index = tds.Hash()
   for index, word in ipairs(word_list) do
      word_index[word] = index
   end

   -- Write it to file
   fd = io.open(list, 'w')
   for index, word in ipairs(word_list) do
      fd:write('"', word:gsub("\n", "\\n"):gsub("\"", "\"\""), '","',
               word_count[word], '","', word_freq[word], '"\n')
   end

   return word_index, #word_list
end

function joe.constructWords(input, output, word_index, word_total)
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

      ofd:write('"', content[1], '"')
      for i = 2, #content do
         content[i] = content[i]:gsub('\\n', '\n'):gsub("^%s*(.-)%s*$", "%1")
         -- All punctuation characters except for hyphen "-"
         content[i] = content[i]:gsub(
            '([!"#$%%&\'()*+,./:;<=>?@%[\\%]^_`{|}~])', ' %1 ')
         local first_write = true
         ofd:write(',"')
         for word in content[i]:gmatch('[%S]+') do
            local index = word_index[word] or word_total + 1
            if first_write then
               first_write = false
               ofd:write(index)
            else
               ofd:write(' ', index)
            end
         end
         ofd:write('"')
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

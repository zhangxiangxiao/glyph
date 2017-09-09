--[[
Construct word bag-of-element format
Copyright 2016 Xiang Zhang

Usage: th construct_wordbag.lua [input] [output] [limit] [replace]
--]]

local io = require('io')
local math = require('math')
local torch = require('torch')

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input = arg[1] or '../data/dianping/train_word.t7b'
   local output = arg[2] or '../data/dianping/train_wordbag.t7b'
   local limit = arg[3] and tonumber(arg[3]) or 200000
   local replace = arg[4] and tonumber(arg[4]) or 200001

   print('Loading data from '..input)
   local data = torch.load(input)

   print('Counting words')
   local count = joe.countBag(data, limit, replace)
   print('Total number of values: '..count)

   print('Constructing word bag data')
   local bag = joe.constructBag(data, count, limit, replace)

   print('Saving to '..output)
   torch.save(output, bag)
end

function joe.countBag(data, limit, replace)
   local code, code_value = data.code, data.code_value

   local count = 0
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
               local word = code_value[code[i][j][k][1] + l - 1]
               if word > limit then
                  word = replace
               end
               if not index[word] then
                  count = count + 1
                  index[word] = 1
               else
                  index[word] = index[word] + 1
               end
            end
         end
      end
      print('\rProcessed texts: '..code[i]:size(1)..'/'..code[i]:size(1))
   end

   return count
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
               local word = code_value[code[i][j][k][1] + l - 1]
               if word > limit then
                  word = replace
               end
               if not index[word] then
                  count = count + 1
                  index[word] = 1
                  pointer[#pointer + 1] = word
               else
                  index[word] = index[word] + 1
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

joe.main()
return joe

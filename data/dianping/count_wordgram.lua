--[[
Parallelized wordgram counting program
Copyright Xiang Zhang 2016

Usage: th count_wordgram.lua [input] [output_prefix] [list] [grams] [chunks]
   [threads] [batch] [buffer]

Comment: This program is a map-reduce like process. During map, each sample is
   separated into character-ngrams. During reduce, these character-ngrams are
   aggregated per-batch samples and output to file chunks. Which files chunk to
   put the gram is determined by a hash value of the gram string, therefore
   instances of the same gram always end up in the same file. This program is
   necessary because a linear aggregation program can easily overflow memory for
   several millions of samples.
--]]

local hash = require('hash')
local io = require('io')
local math = require('math')
local tds = require('tds')
local threads = require('threads')
local torch = require('torch')

local Queue = require('queue')

-- Library configurations
threads.serialization('threads.sharedserialize')

-- A Logic Named Joe
local joe = {}

-- Constant values
joe.SEED = 0

-- Main program entry
function joe.main()
   local input = arg[1] or '../data/dianping/train_word.t7b'
   local output_prefix = arg[2] or '../data/dianping/train_wordgram_count/'
   local list = arg[3] or '../data/dianping/train_word_list.csv'
   local num_grams = arg[4] and tonumber(arg[4]) or 5
   local chunks = arg[5] and tonumber(arg[5]) or 100
   local num_threads = arg[6] and tonumber(arg[6]) or 10
   local batch = arg[7] and tonumber(arg[7]) or 100000
   local buffer = arg[8] and tonumber(arg[8]) or 1000

   print('Loading data from '..input)
   local data = torch.load(input)
   print('Loading list from '..list)
   local freq, word_list = joe.readList(list)
   print('Opening output files with prefix '..output_prefix)
   local fds = {}
   for i = 1, chunks do
      fds[i] = io.open(output_prefix..tostring(i)..'.csv', 'w')
   end
   joe.fds = fds
   print('Setting finished threads to 0')
   joe.finished = 0
   print('Creating record')
   joe.record = tds.Hash()
   print('Setting item counter to 0')
   joe.count = 0
   print('Storing options')
   joe.batch = batch

   print('Creating queues')
   local queue = Queue(buffer)
   print('Creating mutex')
   local mutex = threads.Mutex()
   print('Creating '..num_threads..' threads')
   local init_thread = joe.initThread()
   local block = threads.Threads(num_threads, init_thread)
   block:specific(true)
   print('Deploying thread jobs')
   joe.deployThreads(
      data, word_list, num_grams, queue, mutex, block, num_threads)

   print('Entering main thread loop')
   while joe.finished < num_threads do
      local rpc = queue:pop()
      joe[rpc.func](unpack(rpc.arg))
   end
   if math.fmod(joe.count, batch) ~= 0 then
      print('Writing records to files at '..joe.count)
      joe.writeRecord()
   end

   print('Destroying mutex')
   mutex:free()
   print('Closing files')
   for _, fd in ipairs(fds) do
      fd:close()
   end

   print('Synchronizing and terminating the threads')
   block:synchronize()
   block:terminate()
end

-- Thread initialization callback
function joe.initThread()
   return function ()
      local torch = require('torch')
      local Queue = require('queue')
   end
end

-- Thread job deploying threads
function joe.deployThreads(
      data, word_list, num_grams, queue, mutex, block, num_threads)
   local progress = torch.LongTensor(2)
   progress[1] = 1
   progress[2] = 0
   for i = 1, num_threads do
      print('Deploying job for thread '..i)
      local thread_job = joe.threadJob(
         data, word_list, num_grams, queue, mutex:id(), progress, i)
      block:addjob(i, thread_job)
      local rpc = queue:pop()
      while rpc.func ~= 'notifyDeploy' do
         joe[rpc.func](unpack(rpc.arg))
         rpc = queue:pop()
      end
      print('rpc = notifyDeploy, thread = '..rpc.arg[1])
   end
end

-- Write records to file
function joe.writeRecord()
   for code, item in pairs(joe.record) do
      local chunk = hash.hash(code, joe.SEED, #joe.fds) + 1
      joe.fds[chunk]:write(
         '"', code, '","', item[1]:gsub('\n', '\\n'):gsub('"', '""'), '","',
         item[2], '","', item[3], '"\n')
   end
   joe.record = tds.Hash()
   collectgarbage()
end

-- Thread job
function joe.threadJob(
      data, word_list, num_grams, queue, mutex_id, progress, thread_id)
   local utf8str = joe.utf8str()
   return function()
      local math = require('math')
      local string = require('string')
      local threads = require('threads')
      local mutex = threads.Mutex(mutex_id)

      -- Notify the deployment
      queue:push{func = 'notifyDeploy', arg = {__threadid}}

      local code, code_value = data.code, data.code_value
      local class, item

      -- Obtain next sample
      local function nextSample()
         mutex:lock()
         if code[progress[1]] == nil then
            class = progress[1]
            item = progress[2]
         elseif code[progress[1]]:size(1) < progress[2] + 1 then
            progress[1] = progress[1] + 1
            progress[2] = 1
            class = progress[1]
            item = progress[2]
         else
            progress[2] = progress[2] + 1
            class = progress[1]
            item = progress[2]
         end
         mutex:unlock()
      end

      local n = 0
      nextSample()
      while code[class] ~= nil do
         n = n + 1
         if math.fmod(n, 100) == 0 then
            queue:push{
               func = 'print',
               arg = {__threadid,
                      'Processing class '..class..', item '..item..
                         ', total '..n}}
            collectgarbage()
         end
         local term_count, doc_count = {}, {}
         -- Iterate through the fields
         for i = 1, code[class][item]:size(1) do
            -- Iterate through the grams
            for j = 1, num_grams do
               -- Iterate through the positions
               for k = 1, code[class][item][i][2] - j + 1 do
                  local code_string = tostring(
                     code_value[code[class][item][i][1] + k - 1])
                  for l = 2, j do
                     code_string = code_string..' '..tostring(
                        code_value[code[class][item][i][1] + k - 1 + l - 1])
                  end
                  if not term_count[code_string] then
                     term_count[code_string] = 1
                     doc_count[code_string] = 1
                  else
                     term_count[code_string] = term_count[code_string] + 1
                  end
               end
            end
         end
         -- Compress record to data
         local items = {}
         for code_string, _ in pairs(term_count) do
            local gram_string = ''
            for value in code_string:gmatch('[%S]+') do
               local value = tonumber(value)
               gram_string = gram_string..' '..(word_list[value] or '')
            end
            items[#items + 1] = {
               code_string, gram_string, term_count[code_string],
               doc_count[code_string]}
         end
         -- Send data to record
         queue:push{func = 'recordItem', arg = {__threadid, items}}
         nextSample()
      end

      -- Notify main thread that this thread has ended
      queue:push{func = 'notifyExit', arg = {__threadid}}
   end
end

-- Record item
function joe.recordItem(thread_id, items)
   for _, item in pairs(items) do
      if joe.record[item[1]] then
         joe.record[item[1]][2] = joe.record[item[1]][2] + item[3]
         joe.record[item[1]][3] = joe.record[item[1]][3] + item[4]
      else
         joe.record[item[1]] = tds.Vec{item[2], item[3], item[4]}
      end
   end
   joe.count = joe.count + 1

   -- Check write
   if math.fmod(joe.count, joe.batch) == 0 then
      print('Writing records to files at '..joe.count)
      joe.writeRecord()
   end
end



-- Print information
function joe.print(thread_id, message)
   print('rpc = print, thread = '..thread_id..', message = '..message)
end

-- Notify exit
function joe.notifyExit(thread_id)
   joe.finished = joe.finished + 1
   print('rpc = notifyExit, thread = '..thread_id..
            ', finished = '..joe.finished)
end

-- UTF-8 encoding function
-- Ref: http://stackoverflow.com/questions/7983574/how-to-write-a-unicode-symbol
--      -in-lua
function joe.utf8str()
   local bytemarkers = {{0x7FF, 192}, {0xFFFF, 224}, {0x1FFFFF, 240}}
   return function (decimal)
      local string = require('string')
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
end

function joe.readList(list)
   local freq = {}
   local word_list = tds.Hash()
   local fd = io.open(list)
   for line in fd:lines() do
      local content = joe.parseCSVLine(line)
      content[2] = content[2]:gsub('\\n', '\n')
      freq[#freq + 1] = tonumber(content[3])
      word_list[#freq] = content[1]:gsub('\\n', '\n')
   end
   return torch.Tensor(freq), word_list
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

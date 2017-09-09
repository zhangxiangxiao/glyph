--[[
Create reviews in csv format from original txt file
Copyright 2015-2016 Xiang Zhang

Usage: th construct_reviews [input] [output]
--]]

local cjson = require('cjson')
local io = require('io')
local math = require('math')

local joe = {}

function joe.main()
   local input = arg[1] or '../data/dianping/reviews.txt'
   local output = arg[2] or '../data/dianping/reviews.csv'

   local ifd = io.open(input)
   local ofd = io.open(output, "w")
   local n = 0
   local valid = 0
   for line in ifd:lines() do
      n = n + 1
      if math.fmod(n, 10000) == 0 then
         io.write('\rProcessing line: ', n, ', valid: ', valid)
         io.flush()
      end

      -- Skip the first line
      if n > 1 then
         -- Break content to url and json
         local point = line:find('%^')
         local data = line:sub(point + 2):gsub("^%s*(.-)%s*$", "%1")
         -- Parse the data
         local parsed = cjson.decode(data)
         local content = parsed.content:gsub("^%s*(.-)%s*$", "%1")
         local rate = tonumber(parsed.rate)
         -- Record to csv
         if rate and rate >= 0 and #content > 0 then
            valid = valid + 1
            content = content:gsub("\n", "\\n"):gsub("\"", "\"\"")
            ofd:write('"'..rate..'","'..content..'"\n')
         end
      end
   end
   ifd:close()
   ofd:close()
   print('\rProcessed lines: '..n..', valid: '..valid)
end

joe.main()
return joe
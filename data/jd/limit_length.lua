--[[
Limit length for data
Copyright 2016 Xiang Zhang

Usage: th limit_length.lua [input] [output] [min] [max]
--]]

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input = arg[1] or '../data/jd/sentiment/comment_sorted_nonull.csv'
   local output = arg[2] or '../data/jd/sentiment/comment_sorted_limited.csv'
   local min = tonumber(arg[3] or 0)
   local max = tonumber(arg[4] or math.huge)

   print('Limiting data')
   joe.limit(input, output, min, max)
end

function joe.limit(input, output, min, max)
   local ifd = io.open(input)
   local ofd = io.open(output, 'w')
   local n = 0
   local m = 0
   for line in ifd:lines() do
      n = n + 1

      local content = joe.parseCSVLine(line)
      local length = 0
      for i = 2, #content do
         length = length + content[i]:gsub("^%s*(.-)%s*$", "%1"):len()
      end

      if length >= min and length <= max then
         m = m + 1
         ofd:write(line, '\n')
      end

      if math.fmod(n, 100000) == 0 then
         io.write('\rProcessing line: ', n, ', Saved lines: ', m)
         io.flush()
      end
   end
   print('\rProcessed lines: '..n..', Saved lines: '..m)
   ifd:close()
   ofd:close()
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

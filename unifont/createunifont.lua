--[[
Create unifont database from png file
Copyright 2015 Xiang Zhang

Usage: qlua createunifont.lua [input] [output]
--]]

local image = require('image')
local io = require('io')
local math = require("math")
local torch = require("torch")

-- A Logic Named Joe
local joe = {}

function joe.main()
   local input = arg[1] or 'unifont/unifont-8.0.01.png'
   local output = arg[2] or 'unifont/unifont-8.0.01.t7b'
   local row = arg[3] and tonumber(arg[3]) or 256
   local startx = arg[4] and tonumber(arg[4]) or 33
   local starty = arg[5] and tonumber(arg[5]) or 65
   local width = arg[6] and tonumber(arg[6]) or 16
   local height = arg[7] and tonumber(arg[7]) or width
   local num = arg[8] and tonumber(arg[8]) or 65536

   print('Loading data from '..input)
   local im = image.load(input)
   im = im[1]:double():mul(-1):add(1)
   local data = torch.Tensor(num, height, width)

   for i = 1, num do
      local x = startx + math.fmod(i - 1, row) * width
      local y = starty + math.floor((i - 1)/row) * height
      data[i]:copy(im[{{y, y + height - 1},{x, x + width - 1}}])
      if math.fmod(i, 1000) == 0 then
         io.write('\rProcessing character: '..i..'/'..num)
         joe.win = image.display({image = data[i], win = joe.win, zoom = 8})
      end
   end
   joe.win = image.display({image = data[num], win = joe.win, zoom = 8})
   print('\rProcessed characters: '..num..'/'..num)

   print('Saving to '..output)
   torch.save(output, data)
end

joe.main()

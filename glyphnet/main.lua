--[[
Main program for GlyphNet training
Copyright 2015 Xiang Zhang
--]]

local torch = require('torch')

local Driver = require('driver')

-- A Logic Named Joe
local joe = {}

function joe.main(arg)
   -- Load the configuration
   local config = dofile('config.lua')
   -- Build parameter table based on configuration
   local params = joe.buildArgumentTable(config)
   -- Parse arguments based on configuration
   config = joe.parseArguments(arg, params, config)
   -- Build the driver
   local driver = Driver(config, config.driver)
   -- Start the driver
   driver:run()
end

function joe.buildArgumentTable(config, params, prefix)
   local params = params or {}
   local prefix = prefix or ''
   for key, val in pairs(config) do
      if type(key) == 'string' then
         local val_type = type(val)
         if val_type == 'string' or val_type == 'number' then
            params[prefix..key] = val
         elseif val_type == 'boolean' then
            params[prefix..key] = tostring(val)
         elseif val_type == 'table' then
            params = joe.buildArgumentTable(val, params, prefix..key..'_')
         else
            print('Joe argument '..prefix..key..' type unsupported')
         end
      else
         print('Joe argument key '..prefix..tostring(key)..' not a string')
      end
   end
   return params
end

function joe.parseArguments(arg, params, config)
   local cmd = torch.CmdLine()
   for key, val in pairs(params) do
      cmd:option('-'..key, val)
   end

   local parsed = cmd:parse(arg)
   return joe.parseArgumentTable(config, parsed)
end

function joe.parseArgumentTable(config, params, prefix)
   local prefix = prefix or ''

   for key, val in pairs(config) do
      if type(key) == 'string' then
         local val_type = type(val)
         if val_type == 'string' or val_type == 'number' then
            config[key] = params[prefix..key] or val
         elseif val_type == 'boolean' then
            if params[prefix..key] == 'true' then
               config[key] = true
            elseif params[prefix..key] == 'false' then
               config[key] = false
            else
               error('Argument '..prefix..key..' must be true or false')
            end
         elseif val_type == 'table' then
            config[key] = joe.parseArgumentTable(val, params, prefix..key..'_')
         end
      end
   end
   return config
end

-- Call the main program
joe.main(arg)

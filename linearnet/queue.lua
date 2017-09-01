--[[
Multithreaded queue based on tds
Copyright 2015 Xiang Zhang
--]]

local class = require('pl.class')
local ffi = require('ffi')
local serialize = require('threads.sharedserialize')
local tds = require('tds')
local threads = require('threads')
local torch = require('torch')

-- Append an underscore to distinguish between metatable and class name
local Queue_ = torch.class('Queue')

-- Constructor
-- n: buffer size
function Queue_:__init(size)
   self.data = tds.hash()
   self.pointer = torch.LongTensor(3):fill(1)
   self.pointer[3] = 0
   self.size = size or 10
   self.mutex = threads.Mutex()
   self.added_condition = threads.Condition()
   self.removed_condition = threads.Condition()
end

function Queue_:push(item)
   local storage = serialize.save(item)
   self.mutex:lock()
   while self.pointer[3] == self.size do
      self.removed_condition:wait(self.mutex)
   end
   self.data[self.pointer[1]] = storage:string()
   self.pointer[1] = math.fmod(self.pointer[1], self.size) + 1
   self.pointer[3] = self.pointer[3] + 1
   self.mutex:unlock()
   self.added_condition:signal()
end

function Queue_:pop()
   self.mutex:lock()
   while self.pointer[3] == 0 do
      self.added_condition:wait(self.mutex)
   end
   local storage = torch.CharStorage():string(self.data[self.pointer[2]])
   self.pointer[2] = math.fmod(self.pointer[2], self.size) + 1
   self.pointer[3] = self.pointer[3] - 1
   self.mutex:unlock()
   self.removed_condition:signal()
   local item = serialize.load(storage)
   return item
end

function Queue_:push_async(item)
   if self.pointer[3] == self.size then
      return
   end
   local storage = serialize.save(item)
   self.mutex:lock()
   if self.pointer[3] == self.size then
      self.mutex:unlock()
      return
   end
   self.data[self.pointer[1]] = storage:string()
   self.pointer[1] = math.fmod(self.pointer[1], self.size) + 1
   self.pointer[3] = self.pointer[3] + 1
   self.mutex:unlock()
   self.added_condition:signal()
   return item
end

function Queue_:pop_async()
   if self.pointer[3] == 0 then
      return
   end
   self.mutex:lock()
   if self.pointer[3] == 0 then
      self.mutex:unlock()
      return
   end
   local storage = torch.CharStorage():string(self.data[self.pointer[2]])
   self.pointer[2] = math.fmod(self.pointer[2], self.size) + 1
   self.pointer[3] = self.pointer[3] - 1
   self.mutex:unlock()
   self.removed_condition:signal()
   local item = serialize.load(storage)
   return item
end

function Queue_:free()
   self.mutex:free()
   self.added_condition:free()
   self.removed_condition:free()
end

function Queue_:__write(f)
   local data = self.data
   f:writeLong(torch.pointer(data))
   tds.C.tds_hash_retain(data)

   local pointer = self.pointer
   f:writeLong(torch.pointer(pointer))
   pointer:retain()

   f:writeObject(self.size)
   f:writeObject(self.mutex:id())
   f:writeObject(self.added_condition:id())
   f:writeObject(self.removed_condition:id())
end

function Queue_:__read(f)
   local data = f:readLong()
   data = ffi.cast('tds_hash&', data)
   ffi.gc(data, tds.C.tds_hash_free)
   self.data = data

   local pointer = f:readLong()
   pointer = torch.pushudata(pointer, 'torch.LongTensor')
   self.pointer = pointer
   
   self.size = f:readObject()
   self.mutex = threads.Mutex(f:readObject())
   self.added_condition = threads.Condition(f:readObject())
   self.removed_condition = threads.Condition(f:readObject())
end

-- Return class name, not the underscored metatable
return Queue

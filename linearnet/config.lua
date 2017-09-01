--[[
Configuration for LinearNet
Copyright 2016 Xiang Zhang
--]]

-- Name space
local config = {}

-- Training data configuration
config.train_data = {}
config.train_data.file = 'data/dianping/train_charbag.t7b'

-- Testing data configuration
config.test_data = {}
config.test_data.file = 'data/dianping/test_charbag.t7b'

-- Model configuration
config.model = {}
config.model.size = 200001
config.model.dimension = 2
config.model.decay = 1e-5

-- Trainer configuration
config.train = {}
config.train.rate = 1e-3

-- Tester configuration
config.test = {}

-- Driver configuration
config.driver = {}
config.driver.loss = 'nn.ClassNLLCriterion'
config.driver.threads = 10
config.driver.buffer = 100
config.driver.steps = 100000
config.driver.epoches = 1000
config.driver.interval = 5
config.driver.location = 'models/dianping/charbag'
config.driver.initialization = 1e-2
config.driver.plot = true
config.driver.debug = false
config.driver.resume = false

return config

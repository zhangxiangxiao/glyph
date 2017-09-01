--[[
Configuration for EmbedNet
Copyright Xiang Zhang 2016
--]]

-- Name space
local config = {}

-- Training data configurations
config.train_data = {}
config.train_data.file = 'data/dianping/train_code.t7b'
config.train_data.batch = 16
config.train_data.replace = 65537
config.train_data.shift = 0

-- Testing data configurations
config.test_data = {}
config.test_data.file = 'data/dianping/test_code.t7b'
config.test_data.batch = 16
config.test_data.replace = 65537
config.test_data.shift = 0

-- Model configurations
config.model = {}
config.model.cudnn = true

-- Model variations configuration
config.variation = {}

-- Large model configuration
local embedding = {}
embedding[1] = {name = 'nn.LookupTable', nIndex = 65537, nOutput = 256,
                paddingValue = config.train_data.replace}
embedding[2] = {name = 'nn.Transpose', permutations = {{2, 3}}}
local temporal = {}
temporal[1] = {name = 'nn.TemporalConvolutionMM', inputFrameSize = 256,
               outputFrameSize = 256, kW = 3, dW = 1, padW = 1}
temporal[2] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[3] = {name = 'nn.TemporalConvolutionMM', inputFrameSize = 256,
               outputFrameSize = 256, kW = 3, dW = 1, padW = 1}
temporal[4] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[5] = {name = 'nn.TemporalMaxPoolingMM', kW = 2, dW = 2}
temporal[6] = {name = 'nn.TemporalConvolutionMM', inputFrameSize = 256,
               outputFrameSize = 256, kW = 3, dW = 1, padW = 1}
temporal[7] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[8] = {name = 'nn.TemporalConvolutionMM', inputFrameSize = 256,
               outputFrameSize = 256, kW = 3, dW = 1, padW = 1}
temporal[9] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[10] = {name = 'nn.TemporalMaxPoolingMM', kW = 2, dW = 2}
temporal[11] = {name = 'nn.TemporalConvolutionMM', inputFrameSize = 256,
                outputFrameSize = 256, kW = 3, dW = 1, padW = 1}
temporal[12] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[13] = {name = 'nn.TemporalConvolutionMM', inputFrameSize = 256,
                outputFrameSize = 256, kW = 3, dW = 1, padW = 1}
temporal[14] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[15] = {name = 'nn.TemporalMaxPoolingMM', kW = 2, dW = 2}
temporal[16] = {name = 'nn.TemporalConvolutionMM', inputFrameSize = 256,
                outputFrameSize = 256, kW = 3, dW = 1, padW = 1}
temporal[17] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[18] = {name = 'nn.TemporalConvolutionMM', inputFrameSize = 256,
                outputFrameSize = 256, kW = 3, dW = 1, padW = 1}
temporal[19] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[20] = {name = 'nn.TemporalMaxPoolingMM', kW = 2, dW = 2}
temporal[21] = {name = 'nn.TemporalConvolutionMM', inputFrameSize = 256,
                outputFrameSize = 256, kW = 3, dW = 1, padW = 1}
temporal[22] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[23] = {name = 'nn.TemporalConvolutionMM', inputFrameSize = 256,
                outputFrameSize = 256, kW = 3, dW = 1, padW = 1}
temporal[24] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[25] = {name = 'nn.TemporalMaxPoolingMM', kW = 2, dW = 2}
temporal[26] = {name = 'nn.Reshape', size = 4096, batchMode = true}
temporal[27] = {name = 'nn.Linear', inputSize = 4096, outputSize = 1024}
temporal[28] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[29] = {name = 'nn.Dropout', p = 0.5, v2 = true, inplace = true}
temporal[30] = {name = 'nn.Linear', inputSize = 1024, outputSize = 2}
temporal[31] = {name = 'nn.LogSoftMax'}
config.variation['large'] =
   {embedding = embedding, temporal = temporal, length = 512}

-- Small model configuration
local embedding = {}
embedding[1] = {name = 'nn.LookupTable', nIndex = 65537, nOutput = 256,
                paddingValue = config.train_data.replace}
embedding[2] = {name = 'nn.Transpose', permutations = {{2, 3}}}
local temporal = {}
temporal[1] = {name = 'nn.TemporalConvolutionMM', inputFrameSize = 256,
               outputFrameSize = 256, kW = 3, dW = 1, padW = 1}
temporal[2] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[3] = {name = 'nn.TemporalConvolutionMM', inputFrameSize = 256,
               outputFrameSize = 256, kW = 3, dW = 1, padW = 1}
temporal[4] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[5] = {name = 'nn.TemporalMaxPoolingMM', kW = 3, dW = 3}
temporal[6] = {name = 'nn.TemporalConvolutionMM', inputFrameSize = 256,
               outputFrameSize = 256, kW = 3, dW = 1, padW = 1}
temporal[7] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[8] = {name = 'nn.TemporalConvolutionMM', inputFrameSize = 256,
               outputFrameSize = 256, kW = 3, dW = 1, padW = 1}
temporal[9] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[10] = {name = 'nn.TemporalMaxPoolingMM', kW = 3, dW = 3}
temporal[11] = {name = 'nn.TemporalConvolutionMM', inputFrameSize = 256,
                outputFrameSize = 256, kW = 3, dW = 1, padW = 1}
temporal[12] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[13] = {name = 'nn.TemporalConvolutionMM', inputFrameSize = 256,
                outputFrameSize = 256, kW = 3, dW = 1, padW = 1}
temporal[14] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[15] = {name = 'nn.TemporalMaxPoolingMM', kW = 3, dW = 3}
temporal[16] = {name = 'nn.Reshape', size = 4608, batchMode = true}
temporal[17] = {name = 'nn.Linear', inputSize = 4608, outputSize = 1024}
temporal[18] = {name = 'nn.Threshold', th = 1e-6, v = 0, ip = true}
temporal[19] = {name = 'nn.Dropout', p = 0.5, v2 = true, inplace = true}
temporal[20] = {name = 'nn.Linear', inputSize = 1024, outputSize = 2}
temporal[21] = {name = 'nn.LogSoftMax'}
config.variation['small'] =
   {embedding = embedding, temporal = temporal, length = 486}

-- Trainer settings
config.train = {}
config.train.momentum = 0.9
config.train.decay = 1e-5
-- These are just multipliers to config.driver.rate
-- For every config.driver.schedule * config.driver.steps
config.train.rates =
   {1/1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256, 1/512, 1/1024}

-- Tester settings
config.test = {}

-- Visualizer settings
config.visualizer = {}
config.visualizer.width = 1200
config.visualizer.scale = 4
config.visualizer.height = 64

-- Driver configurations
config.driver = {}
config.driver.type = 'torch.CudaTensor'
config.driver.device = 1
config.driver.loss = 'nn.ClassNLLCriterion'
config.driver.variation = 'large'
config.driver.dimension = 65537
config.driver.steps = 100000
config.driver.epoches = 100
config.driver.schedule = 8
config.driver.rate = 1e-5
config.driver.interval = 5
config.driver.location = 'models/dianping/temporal12length512feature256'
config.driver.plot = true
config.driver.visualize = true
config.driver.debug = false
config.driver.resume = false

-- Main configuration
config.joe = {}

return config

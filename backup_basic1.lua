-- Modification from the codebase of scott's icml16
-- please check https://github.com/reedscot/icml2016 for details

require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'lfs'
require 'torch'
require "orbit"

local orbit = require("orbit");

-- declaration
module("get_embedding", package.seeall, orbit.new)

torch.setdefaulttensortype('torch.FloatTensor')

local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
    dict[alphabet:sub(i,i)] = i
end
ivocab = {}
for k,v in pairs(dict) do
  ivocab[v] = k
end

opt = {
  filenames = '/data2/adsue/caption_data/mini_batch_captions.t7',
  doc_length = 201,
  net_txt = '/data2/adsue/pretrained/coco_gru18_bs64_cls0.5_ngf128_ndf128_a10_c512_80_net_T.t7',
  queries = '/data2/adsue/caption_data/mini_batch_captions.txt'
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

net_txt = torch.load(opt.net_txt)
if net_txt.protos ~=nil then net_txt = net_txt.protos.enc_doc end


net_txt:evaluate()

local opt = opt
local net_txt = net_txt

-- basic1.lua
require"orbit"

local orbit = require"orbit"

-- declaration
module("basic1", package.seeall, orbit.new)

-- handler
function index(web)
  return render_index(web)
end

-- dispatch
basic1:dispatch_get(index, "/", "/index")

-- render
function render_index(web)
  -- Extract all text features.
  local fea_txt = {}
  -- Decode text for sanity check.
  local raw_txt = {}
  local raw_img = {}
  
  for query_str in io.lines(opt.queries) do
    local txt = torch.zeros(1,opt.doc_length,#alphabet)
    for t = 1,opt.doc_length do
      local ch = query_str:sub(t,t)
      local ix = dict[ch]
      if ix ~= 0 and ix ~= nil then
        txt[{1,t,ix}] = 1
      end
    end
    raw_txt[#raw_txt+1] = query_str
    txt = txt:cuda()

    fea_txt[#fea_txt+1] = net_txt:forward(txt):clone()
  end

  torch.save(opt.filenames, {raw_txt=raw_txt, fea_txt=fea_txt})
  return 'done'
end

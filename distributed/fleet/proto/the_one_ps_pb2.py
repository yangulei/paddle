# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: paddle/fluid/distributed/the_one_ps.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n)paddle/fluid/distributed/the_one_ps.proto\x12\x12paddle.distributed\"\xe1\x01\n\x11\x46sClientParameter\x12\x46\n\x07\x66s_type\x18\x01 \x01(\x0e\x32/.paddle.distributed.FsClientParameter.FsApiType:\x04HDFS\x12\x0b\n\x03uri\x18\x02 \x01(\t\x12\x0c\n\x04user\x18\x03 \x01(\t\x12\x0e\n\x06passwd\x18\x04 \x01(\t\x12\x13\n\x0b\x62uffer_size\x18\x05 \x01(\x05\x12\x12\n\nhadoop_bin\x18\x33 \x01(\t\x12\x10\n\x08\x61\x66s_conf\x18\x65 \x01(\t\"\x1e\n\tFsApiType\x12\x08\n\x04HDFS\x10\x00\x12\x07\n\x03\x41\x46S\x10\x01\"\xe5\x02\n\x0bPSParameter\x12\x14\n\x0cworker_class\x18\x01 \x01(\t\x12\x14\n\x0cserver_class\x18\x02 \x01(\t\x12\x16\n\x0einstance_class\x18\x03 \x01(\t\x12\x15\n\x0binit_gflags\x18\x04 \x01(\t:\x00\x12\x39\n\x0cworker_param\x18\x65 \x01(\x0b\x32#.paddle.distributed.WorkerParameter\x12\x39\n\x0cserver_param\x18\x66 \x01(\x0b\x32#.paddle.distributed.ServerParameter\x12\x44\n\rtrainer_param\x18\xad\x02 \x03(\x0b\x32,.paddle.distributed.DownpourTrainerParameter\x12?\n\x0f\x66s_client_param\x18\xf5\x03 \x01(\x0b\x32%.paddle.distributed.FsClientParameter\"]\n\x0fWorkerParameter\x12J\n\x15\x64ownpour_worker_param\x18\x01 \x01(\x0b\x32+.paddle.distributed.DownpourWorkerParameter\"[\n\x17\x44ownpourWorkerParameter\x12@\n\x14\x64ownpour_table_param\x18\x01 \x03(\x0b\x32\".paddle.distributed.TableParameter\"\x9e\x01\n\x17\x44ownpourServerParameter\x12@\n\x14\x64ownpour_table_param\x18\x01 \x03(\x0b\x32\".paddle.distributed.TableParameter\x12\x41\n\rservice_param\x18\x02 \x01(\x0b\x32*.paddle.distributed.ServerServiceParameter\"]\n\x0fServerParameter\x12J\n\x15\x64ownpour_server_param\x18\x01 \x01(\x0b\x32+.paddle.distributed.DownpourServerParameter\"\xa1\x02\n\x18\x44ownpourTrainerParameter\x12<\n\x0b\x64\x65nse_table\x18\x01 \x03(\x0b\x32\'.paddle.distributed.DenseTableParameter\x12>\n\x0csparse_table\x18\x02 \x03(\x0b\x32(.paddle.distributed.SparseTableParameter\x12\x1d\n\x15push_sparse_per_batch\x18\x03 \x01(\x05\x12\x1c\n\x14push_dense_per_batch\x18\x04 \x01(\x05\x12\x0f\n\x07skip_op\x18\x05 \x03(\t\x12\x39\n\x0eprogram_config\x18\x06 \x03(\x0b\x32!.paddle.distributed.ProgramConfig\"{\n\x13\x44\x65nseTableParameter\x12\x10\n\x08table_id\x18\x01 \x01(\x05\x12\x1b\n\x13\x64\x65nse_variable_name\x18\x02 \x03(\t\x12$\n\x1c\x64\x65nse_gradient_variable_name\x18\x03 \x03(\t\x12\x0f\n\x07\x66\x65\x61_dim\x18\x04 \x01(\x05\"z\n\x14SparseTableParameter\x12\x10\n\x08table_id\x18\x01 \x01(\x05\x12\x13\n\x0b\x66\x65\x61ture_dim\x18\x02 \x01(\x05\x12\x10\n\x08slot_key\x18\x03 \x03(\t\x12\x12\n\nslot_value\x18\x04 \x03(\t\x12\x15\n\rslot_gradient\x18\x05 \x03(\t\"\xc3\x01\n\x16ServerServiceParameter\x12\"\n\x0cserver_class\x18\x01 \x01(\t:\x0c\x42rpcPsServer\x12\"\n\x0c\x63lient_class\x18\x02 \x01(\t:\x0c\x42rpcPsClient\x12$\n\rservice_class\x18\x03 \x01(\t:\rBrpcPsService\x12\x1c\n\x11start_server_port\x18\x04 \x01(\r:\x01\x30\x12\x1d\n\x11server_thread_num\x18\x05 \x01(\r:\x02\x31\x32\"\x99\x01\n\rProgramConfig\x12\x12\n\nprogram_id\x18\x01 \x02(\t\x12\x1c\n\x14push_sparse_table_id\x18\x02 \x03(\x05\x12\x1b\n\x13push_dense_table_id\x18\x03 \x03(\x05\x12\x1c\n\x14pull_sparse_table_id\x18\x04 \x03(\x05\x12\x1b\n\x13pull_dense_table_id\x18\x05 \x03(\x05\"\xc9\x04\n\x0eTableParameter\x12\x10\n\x08table_id\x18\x01 \x01(\x04\x12\x13\n\x0btable_class\x18\x02 \x01(\t\x12\x17\n\tshard_num\x18\x03 \x01(\x04:\x04\x31\x30\x30\x30\x12<\n\x08\x61\x63\x63\x65ssor\x18\x04 \x01(\x0b\x32*.paddle.distributed.TableAccessorParameter\x12;\n\x06tensor\x18\x05 \x01(\x0b\x32+.paddle.distributed.TensorAccessorParameter\x12;\n\x06\x63ommon\x18\x06 \x01(\x0b\x32+.paddle.distributed.CommonAccessorParameter\x12+\n\x04type\x18\x07 \x01(\x0e\x32\x1d.paddle.distributed.TableType\x12\x1e\n\x10\x63ompress_in_save\x18\x08 \x01(\x08:\x04true\x12;\n\x0fgraph_parameter\x18\t \x01(\x0b\x32\".paddle.distributed.GraphParameter\x12\'\n\x19\x65nable_sparse_table_cache\x18\n \x01(\x08:\x04true\x12(\n\x17sparse_table_cache_rate\x18\x0b \x01(\x01:\x07\x30.00055\x12\'\n\x1bsparse_table_cache_file_num\x18\x0c \x01(\r:\x02\x31\x36\x12\x1c\n\renable_revert\x18\r \x01(\x08:\x05\x66\x61lse\x12\x1b\n\x10shard_merge_rate\x18\x0e \x01(\x02:\x01\x31\"\xea\x03\n\x16TableAccessorParameter\x12\x16\n\x0e\x61\x63\x63\x65ssor_class\x18\x01 \x01(\t\x12\x13\n\x07\x66\x65\x61_dim\x18\x04 \x01(\r:\x02\x31\x31\x12\x15\n\nembedx_dim\x18\x05 \x01(\r:\x01\x38\x12\x1c\n\x10\x65mbedx_threshold\x18\x06 \x01(\r:\x02\x31\x30\x12\x44\n\x12\x63tr_accessor_param\x18\x07 \x01(\x0b\x32(.paddle.distributed.CtrAccessorParameter\x12Q\n\x19table_accessor_save_param\x18\x08 \x03(\x0b\x32..paddle.distributed.TableAccessorSaveParameter\x12I\n\x0f\x65mbed_sgd_param\x18\n \x01(\x0b\x32\x30.paddle.distributed.SparseCommonSGDRuleParameter\x12J\n\x10\x65mbedx_sgd_param\x18\x0b \x01(\x0b\x32\x30.paddle.distributed.SparseCommonSGDRuleParameter\x12>\n\x0fgraph_sgd_param\x18\x0c \x01(\x0b\x32%.paddle.distributed.GraphSGDParameter\"S\n\x11GraphSGDParameter\x12\x19\n\x0bnodeid_slot\x18\x01 \x01(\r:\x04\x39\x30\x30\x38\x12#\n\x15\x66\x65\x61ture_learning_rate\x18\x02 \x01(\x02:\x04\x30.05\"\xfe\x02\n\x14\x43trAccessorParameter\x12\x19\n\x0cnonclk_coeff\x18\x01 \x01(\x02:\x03\x30.1\x12\x16\n\x0b\x63lick_coeff\x18\x02 \x01(\x02:\x01\x31\x12\x1b\n\x0e\x62\x61se_threshold\x18\x03 \x01(\x02:\x03\x31.5\x12\x1d\n\x0f\x64\x65lta_threshold\x18\x04 \x01(\x02:\x04\x30.25\x12\x1b\n\x0f\x64\x65lta_keep_days\x18\x05 \x01(\x02:\x02\x31\x36\x12#\n\x15show_click_decay_rate\x18\x06 \x01(\x02:\x04\x30.98\x12\x1d\n\x10\x64\x65lete_threshold\x18\x07 \x01(\x02:\x03\x30.8\x12$\n\x18\x64\x65lete_after_unseen_days\x18\x08 \x01(\x02:\x02\x33\x30\x12\"\n\x17ssd_unseenday_threshold\x18\t \x01(\x05:\x01\x31\x12\x18\n\nshow_scale\x18\n \x01(\x08:\x04true\x12\x17\n\tzero_init\x18\x0b \x01(\x08:\x04true\x12\x19\n\x11load_filter_slots\x18\x0c \x03(\x02\"\x99\x01\n\x17TensorAccessorParameter\x12\x15\n\rfeed_var_name\x18\x01 \x01(\t\x12\x16\n\x0e\x66\x65tch_var_name\x18\x02 \x01(\t\x12\x1a\n\x12startup_program_id\x18\x03 \x01(\x03\x12\x17\n\x0fmain_program_id\x18\x04 \x01(\x03\x12\x1a\n\x12tensor_table_class\x18\x06 \x01(\t\"\xe9\x01\n\x17\x43ommonAccessorParameter\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x12\n\ntable_name\x18\x02 \x01(\t\x12\x12\n\nattributes\x18\x03 \x03(\t\x12\x0e\n\x06params\x18\x04 \x03(\t\x12\x0c\n\x04\x64ims\x18\x05 \x03(\r\x12\x14\n\x0cinitializers\x18\x06 \x03(\t\x12\r\n\x05\x65ntry\x18\x07 \x01(\t\x12\x13\n\x0btrainer_num\x18\x08 \x01(\x05\x12\x0c\n\x04sync\x18\t \x01(\x08\x12\x11\n\ttable_num\x18\n \x01(\r\x12\x11\n\ttable_dim\x18\x0b \x01(\r\x12\x0c\n\x04\x61ttr\x18\x0c \x01(\t\"S\n\x1aTableAccessorSaveParameter\x12\r\n\x05param\x18\x01 \x01(\r\x12\x11\n\tconverter\x18\x02 \x01(\t\x12\x13\n\x0b\x64\x65\x63onverter\x18\x03 \x01(\t\"\xea\x01\n\x1cSparseCommonSGDRuleParameter\x12\x0c\n\x04name\x18\x01 \x01(\t\x12>\n\x05naive\x18\x02 \x01(\x0b\x32/.paddle.distributed.SparseNaiveSGDRuleParameter\x12\x42\n\x07\x61\x64\x61grad\x18\x03 \x01(\x0b\x32\x31.paddle.distributed.SparseAdagradSGDRuleParameter\x12\x38\n\x04\x61\x64\x61m\x18\x04 \x01(\x0b\x32*.paddle.distributed.SparseAdamSGDParameter\"p\n\x1bSparseNaiveSGDRuleParameter\x12\x1b\n\rlearning_rate\x18\x01 \x01(\x01:\x04\x30.05\x12\x1d\n\rinitial_range\x18\x02 \x01(\x01:\x06\x30.0001\x12\x15\n\rweight_bounds\x18\x03 \x03(\x02\"\x8c\x01\n\x1dSparseAdagradSGDRuleParameter\x12\x1b\n\rlearning_rate\x18\x01 \x01(\x01:\x04\x30.05\x12\x18\n\rinitial_g2sum\x18\x02 \x01(\x01:\x01\x33\x12\x1d\n\rinitial_range\x18\x03 \x01(\x01:\x06\x30.0001\x12\x15\n\rweight_bounds\x18\x04 \x03(\x02\"\xc8\x01\n\x16SparseAdamSGDParameter\x12\x1c\n\rlearning_rate\x18\x01 \x01(\x01:\x05\x30.001\x12\x1d\n\rinitial_range\x18\x02 \x01(\x01:\x06\x30.0001\x12\x1d\n\x10\x62\x65ta1_decay_rate\x18\x03 \x01(\x01:\x03\x30.9\x12\x1f\n\x10\x62\x65ta2_decay_rate\x18\x04 \x01(\x01:\x05\x30.999\x12\x1a\n\x0b\x61\x64\x61_epsilon\x18\x05 \x01(\x01:\x05\x31\x65-08\x12\x15\n\rweight_bounds\x18\x06 \x03(\x02\"\xe0\x02\n\x0eGraphParameter\x12\x1a\n\x0etask_pool_size\x18\x01 \x01(\x05:\x02\x32\x34\x12\x12\n\nedge_types\x18\x02 \x03(\t\x12\x12\n\nnode_types\x18\x03 \x03(\t\x12\x18\n\tuse_cache\x18\x04 \x01(\x08:\x05\x66\x61lse\x12 \n\x10\x63\x61\x63he_size_limit\x18\x05 \x01(\x05:\x06\x31\x30\x30\x30\x30\x30\x12\x14\n\tcache_ttl\x18\x06 \x01(\x05:\x01\x35\x12\x37\n\rgraph_feature\x18\x07 \x03(\x0b\x32 .paddle.distributed.GraphFeature\x12\x14\n\ntable_name\x18\x08 \x01(\t:\x00\x12\x14\n\ntable_type\x18\t \x01(\t:\x00\x12\x16\n\tshard_num\x18\n \x01(\x05:\x03\x31\x32\x37\x12\x17\n\x0csearch_level\x18\x0b \x01(\x05:\x01\x31\x12\"\n\x14\x62uild_sampler_on_cpu\x18\x0c \x01(\x08:\x04true\":\n\x0cGraphFeature\x12\x0c\n\x04name\x18\x01 \x03(\t\x12\r\n\x05\x64type\x18\x02 \x03(\t\x12\r\n\x05shape\x18\x03 \x03(\x05\"y\n\x0b\x46LParameter\x12\x33\n\x0b\x66l_strategy\x18\x01 \x01(\x0b\x32\x1e.paddle.distributed.FLStrategy\x12\x35\n\x0b\x63lient_info\x18\x02 \x01(\x0b\x32 .paddle.distributed.FLClientInfo\"g\n\nFLStrategy\x12\x15\n\riteration_num\x18\x01 \x01(\x04\x12\x11\n\tclient_id\x18\x02 \x01(\x04\x12\x18\n\nnext_state\x18\x03 \x01(\t:\x04JOIN\x12\x15\n\x0binit_gflags\x18\x04 \x01(\t:\x00\"\xc2\x01\n\x0c\x46LClientInfo\x12\x11\n\tclient_id\x18\x01 \x01(\r\x12\x13\n\x0b\x64\x65vice_type\x18\x02 \x01(\t\x12\x18\n\x10\x63ompute_capacity\x18\x03 \x01(\x05\x12\x11\n\tbandwidth\x18\x04 \x01(\x05\x12\x46\n\x15local_training_result\x18\x05 \x01(\x0b\x32\'.paddle.distributed.LocalTrainingResult\x12\x15\n\x0binit_gflags\x18\x06 \x01(\t:\x00\"0\n\x13LocalTrainingResult\x12\x0b\n\x03\x61\x63\x63\x18\x01 \x01(\x01\x12\x0c\n\x04loss\x18\x02 \x01(\x01*H\n\tTableType\x12\x13\n\x0fPS_SPARSE_TABLE\x10\x00\x12\x12\n\x0ePS_DENSE_TABLE\x10\x01\x12\x12\n\x0ePS_OTHER_TABLE\x10\x02\x42\x06\x80\x01\x01\xf8\x01\x01')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'paddle.fluid.distributed.the_one_ps_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\200\001\001\370\001\001'
  _TABLETYPE._serialized_start=5607
  _TABLETYPE._serialized_end=5679
  _FSCLIENTPARAMETER._serialized_start=66
  _FSCLIENTPARAMETER._serialized_end=291
  _FSCLIENTPARAMETER_FSAPITYPE._serialized_start=261
  _FSCLIENTPARAMETER_FSAPITYPE._serialized_end=291
  _PSPARAMETER._serialized_start=294
  _PSPARAMETER._serialized_end=651
  _WORKERPARAMETER._serialized_start=653
  _WORKERPARAMETER._serialized_end=746
  _DOWNPOURWORKERPARAMETER._serialized_start=748
  _DOWNPOURWORKERPARAMETER._serialized_end=839
  _DOWNPOURSERVERPARAMETER._serialized_start=842
  _DOWNPOURSERVERPARAMETER._serialized_end=1000
  _SERVERPARAMETER._serialized_start=1002
  _SERVERPARAMETER._serialized_end=1095
  _DOWNPOURTRAINERPARAMETER._serialized_start=1098
  _DOWNPOURTRAINERPARAMETER._serialized_end=1387
  _DENSETABLEPARAMETER._serialized_start=1389
  _DENSETABLEPARAMETER._serialized_end=1512
  _SPARSETABLEPARAMETER._serialized_start=1514
  _SPARSETABLEPARAMETER._serialized_end=1636
  _SERVERSERVICEPARAMETER._serialized_start=1639
  _SERVERSERVICEPARAMETER._serialized_end=1834
  _PROGRAMCONFIG._serialized_start=1837
  _PROGRAMCONFIG._serialized_end=1990
  _TABLEPARAMETER._serialized_start=1993
  _TABLEPARAMETER._serialized_end=2578
  _TABLEACCESSORPARAMETER._serialized_start=2581
  _TABLEACCESSORPARAMETER._serialized_end=3071
  _GRAPHSGDPARAMETER._serialized_start=3073
  _GRAPHSGDPARAMETER._serialized_end=3156
  _CTRACCESSORPARAMETER._serialized_start=3159
  _CTRACCESSORPARAMETER._serialized_end=3541
  _TENSORACCESSORPARAMETER._serialized_start=3544
  _TENSORACCESSORPARAMETER._serialized_end=3697
  _COMMONACCESSORPARAMETER._serialized_start=3700
  _COMMONACCESSORPARAMETER._serialized_end=3933
  _TABLEACCESSORSAVEPARAMETER._serialized_start=3935
  _TABLEACCESSORSAVEPARAMETER._serialized_end=4018
  _SPARSECOMMONSGDRULEPARAMETER._serialized_start=4021
  _SPARSECOMMONSGDRULEPARAMETER._serialized_end=4255
  _SPARSENAIVESGDRULEPARAMETER._serialized_start=4257
  _SPARSENAIVESGDRULEPARAMETER._serialized_end=4369
  _SPARSEADAGRADSGDRULEPARAMETER._serialized_start=4372
  _SPARSEADAGRADSGDRULEPARAMETER._serialized_end=4512
  _SPARSEADAMSGDPARAMETER._serialized_start=4515
  _SPARSEADAMSGDPARAMETER._serialized_end=4715
  _GRAPHPARAMETER._serialized_start=4718
  _GRAPHPARAMETER._serialized_end=5070
  _GRAPHFEATURE._serialized_start=5072
  _GRAPHFEATURE._serialized_end=5130
  _FLPARAMETER._serialized_start=5132
  _FLPARAMETER._serialized_end=5253
  _FLSTRATEGY._serialized_start=5255
  _FLSTRATEGY._serialized_end=5358
  _FLCLIENTINFO._serialized_start=5361
  _FLCLIENTINFO._serialized_end=5555
  _LOCALTRAININGRESULT._serialized_start=5557
  _LOCALTRAININGRESULT._serialized_end=5605
# @@protoc_insertion_point(module_scope)

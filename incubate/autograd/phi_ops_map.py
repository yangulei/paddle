op_map = {
    "abs": {
        "phi_name": "abs",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "accuracy": {
        "phi_name": "accuracy",
        "inputs": {
            "x": "Out",
            "indices": "Indices",
            "label": "Label"
        },
        "outputs": {
            "accuracy": "Accuracy",
            "correct": "Correct",
            "total": "Total"
        }
    },
    "acos": {
        "phi_name": "acos",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "acosh": {
        "phi_name": "acosh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "elementwise_add": {
        "phi_name": "add"
    },
    "addmm": {
        "phi_name": "addmm",
        "inputs": {
            "input": "Input",
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "alpha": "Alpha",
            "beta": "Beta"
        }
    },
    "affine_grid": {
        "phi_name": "affine_grid"
    },
    "reduce_all": {
        "phi_name": "all",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim",
            "keepdim": "keep_dim"
        },
        "int_array": {
            "axis": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "allclose": {
        "phi_name": "allclose",
        "inputs": {
            "x": "Input",
            "y": "Other"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "rtol": {
                "data_type": "std::string",
                "tensor_name": "Rtol"
            },
            "atol": {
                "data_type": "std::string",
                "tensor_name": "Atol"
            }
        }
    },
    "angle": {
        "phi_name": "angle",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "range": {
        "phi_name": "arange",
        "inputs": {
            "start": "Start",
            "end": "End",
            "step": "Step"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "argsort": {
        "phi_name": "argsort",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "indices": "Indices"
        }
    },
    "as_complex": {
        "phi_name": "as_complex",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "as_real": {
        "phi_name": "as_real",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "asin": {
        "phi_name": "asin",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "asinh": {
        "phi_name": "asinh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "assign": {
        "phi_name": "assign",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "atan": {
        "phi_name": "atan",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "atan2": {
        "phi_name": "atan2",
        "inputs": {
            "x": "X1",
            "y": "X2"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "atanh": {
        "phi_name": "atanh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "auc": {
        "phi_name": "auc",
        "inputs": {
            "x": "Predict",
            "label": "Label",
            "stat_pos": "StatPos",
            "stat_neg": "StatNeg",
            "ins_tag_weight": "InsTagWeight"
        },
        "outputs": {
            "auc": "AUC",
            "stat_pos_out": "StatPosOut",
            "stat_neg_out": "StatNegOut"
        }
    },
    "batch_norm": {
        "phi_name": "batch_norm",
        "inputs": {
            "x": "X",
            "mean": "Mean",
            "variance": "Variance",
            "scale": "Scale",
            "bias": "Bias"
        },
        "outputs": {
            "out": "Y",
            "mean_out": "MeanOut",
            "variance_out": "VarianceOut",
            "saved_mean": "SavedMean",
            "saved_variance": "SavedVariance",
            "reserve_space": "ReserveSpace"
        }
    },
    "bce_loss": {
        "phi_name": "bce_loss",
        "inputs": {
            "input": "X",
            "label": "Label"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "bernoulli": {
        "phi_name": "bernoulli",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "bicubic_interp_v2": {
        "phi_name": "bicubic_interp",
        "inputs": {
            "x": "X",
            "out_size": "OutSize",
            "size_tensor": "SizeTensor",
            "scale_tensor": "Scale"
        },
        "outputs": {
            "output": "Out"
        }
    },
    "bilinear_interp_v2": {
        "phi_name": "bilinear_interp",
        "inputs": {
            "x": "X",
            "out_size": "OutSize",
            "size_tensor": "SizeTensor",
            "scale_tensor": "Scale"
        },
        "outputs": {
            "output": "Out"
        }
    },
    "bitwise_and": {
        "phi_name": "bitwise_and",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "bitwise_not": {
        "phi_name": "bitwise_not",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "bitwise_or": {
        "phi_name": "bitwise_or",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "bitwise_xor": {
        "phi_name": "bitwise_xor",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "bmm": {
        "phi_name": "bmm",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "broadcast_tensors": {
        "phi_name": "broadcast_tensors",
        "inputs": {
            "input": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "ceil": {
        "phi_name": "ceil",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "celu": {
        "phi_name": "celu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "cholesky": {
        "phi_name": "cholesky",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "cholesky_solve": {
        "phi_name": "cholesky_solve",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "clip": {
        "phi_name": "clip",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "min": {
                "data_type": "float",
                "tensor_name": "Min"
            },
            "max": {
                "data_type": "float",
                "tensor_name": "Max"
            }
        }
    },
    "complex": {
        "phi_name": "complex",
        "inputs": {
            "real": "X",
            "imag": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "concat": {
        "phi_name": "concat",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "axis"
        },
        "scalar": {
            "axis": {
                "data_type": "int",
                "tensor_name": "AxisTensor"
            }
        }
    },
    "conditional_block": {
        "phi_name": "conditional_block"
    },
    "conj": {
        "phi_name": "conj",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "conv2d": {
        "phi_name": "conv2d"
    },
    "conv2d_fusion": {
        "phi_name": "conv2d_fusion"
    },
    "conv2d_transpose": {
        "phi_name": "conv2d_transpose"
    },
    "conv3d": {
        "phi_name": "conv3d"
    },
    "conv3d_transpose": {
        "phi_name": "conv3d_transpose"
    },
    "cos": {
        "phi_name": "cos",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "cosh": {
        "phi_name": "cosh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "crop_tensor": {
        "phi_name": "crop",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "shape": {
                "data_type": "int",
                "tensor_name": "Shape",
                "tensors_name": "ShapeTensor"
            },
            "offsets": {
                "data_type": "int",
                "tensor_name": "Offsets",
                "tensors_name": "OffsetsTensor"
            }
        }
    },
    "cross": {
        "phi_name": "cross",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim"
        }
    },
    "cumprod": {
        "phi_name": "cumprod",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "dim": "dim"
        }
    },
    "cumsum": {
        "phi_name": "cumsum",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "axis": {
                "data_type": "int",
                "tensor_name": "AxisTensor"
            }
        }
    },
    "data_norm": {
        "phi_name": "data_norm"
    },
    "depthwise_conv2d": {
        "phi_name": "depthwise_conv2d"
    },
    "depthwise_conv2d_transpose": {
        "phi_name": "depthwise_conv2d_transpose"
    },
    "dequantize_linear": {
        "phi_name": "dequantize_linear"
    },
    "determinant": {
        "phi_name": "det",
        "inputs": {
            "x": "Input"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "diag_v2": {
        "phi_name": "diag",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "diag_embed": {
        "phi_name": "diag_embed",
        "inputs": {
            "input": "Input"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "diagonal": {
        "phi_name": "diagonal",
        "inputs": {
            "x": "Input"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "digamma": {
        "phi_name": "digamma",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "dirichlet": {
        "phi_name": "dirichlet",
        "inputs": {
            "alpha": "Alpha"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "dist": {
        "phi_name": "dist",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "distributed_push_sparse": {
        "phi_name": "distributed_push_sparse"
    },
    "elementwise_div": {
        "phi_name": "divide",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "dot": {
        "phi_name": "dot",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "dropout": {
        "phi_name": "dropout",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "mask": "Mask"
        },
        "attrs": {
            "p": "dropout_prob",
            "is_test": "is_test",
            "mode": "dropout_implementation",
            "seed": "seed",
            "fix_seed": "fix_seed"
        }
    },
    "dropout_nd": {
        "phi_name": "dropout_nd"
    },
    "eig": {
        "phi_name": "eig",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out_w": "Eigenvalues",
            "out_v": "Eigenvectors"
        }
    },
    "eigh": {
        "phi_name": "eigh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out_w": "Eigenvalues",
            "out_v": "Eigenvectors"
        }
    },
    "eigvals": {
        "phi_name": "eigvals",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "elementwise_pow": {
        "phi_name": "elementwise_pow"
    },
    "elu": {
        "phi_name": "elu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "lookup_table_v2": {
        "phi_name": "embedding",
        "inputs": {
            "x": "Ids",
            "weight": "W"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "equal": {
        "phi_name": "equal",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "equal_all": {
        "phi_name": "equal_all",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "erf": {
        "phi_name": "erf",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "erfinv": {
        "phi_name": "erfinv",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "exp": {
        "phi_name": "exp",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "expand_v2": {
        "phi_name": "expand",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "shape": "shape"
        },
        "int_array": {
            "shape": {
                "data_type": "int",
                "tensor_name": "Shape",
                "tensors_name": "expand_shapes_tensor"
            }
        }
    },
    "expand_as_v2": {
        "phi_name": "expand_as",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "expm1": {
        "phi_name": "expm1",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "eye": {
        "phi_name": "eye",
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "num_rows": {
                "support_tensor": "True"
            },
            "num_columns": {
                "support_tensor": "True"
            }
        }
    },
    "fake_channel_wise_quantize_abs_max": {
        "phi_name": "fake_channel_wise_quantize_abs_max"
    },
    "fake_channel_wise_quantize_dequantize_abs_max": {
        "phi_name": "fake_channel_wise_quantize_dequantize_abs_max"
    },
    "fake_quantize_abs_max": {
        "phi_name": "fake_quantize_abs_max"
    },
    "fake_quantize_dequantize_abs_max": {
        "phi_name": "fake_quantize_dequantize_abs_max"
    },
    "fake_quantize_dequantize_moving_average_abs_max": {
        "phi_name": "fake_quantize_dequantize_moving_average_abs_max"
    },
    "fake_quantize_moving_average_abs_max": {
        "phi_name": "fake_quantize_moving_average_abs_max"
    },
    "fake_quantize_range_abs_max": {
        "phi_name": "fake_quantize_range_abs_max"
    },
    "fft_c2c": {
        "phi_name": "fft_c2c",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fft_c2r": {
        "phi_name": "fft_c2r",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fft_r2c": {
        "phi_name": "fft_r2c",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fill_diagonal": {
        "phi_name": "fill_diagonal",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "fill_diagonal_tensor": {
        "phi_name": "fill_diagonal_tensor",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "flatten_contiguous_range": {
        "phi_name": "flatten",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "xshape": "XShape"
        },
        "attrs": {
            "start_axis": "start_axis",
            "stop_axis": "stop_axis"
        }
    },
    "flip": {
        "phi_name": "flip",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "floor": {
        "phi_name": "floor",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "elementwise_floordiv": {
        "phi_name": "floor_divide"
    },
    "elementwise_fmax": {
        "phi_name": "fmax"
    },
    "elementwise_fmin": {
        "phi_name": "fmin"
    },
    "fold": {
        "phi_name": "fold",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Y"
        }
    },
    "frame": {
        "phi_name": "frame",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "frobenius_norm": {
        "phi_name": "frobenius_norm",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim",
            "keepdim": "keep_dim"
        },
        "int_array": {
            "axis": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "fill_constant": {
        "phi_name": "full"
    },
    "fill_any_like": {
        "phi_name": "full_like",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "value": "value",
            "dtype": "dtype"
        }
    },
    "fused_conv2d": {
        "phi_name": "fused_conv2d"
    },
    "fused_transpose": {
        "phi_name": "fused_transpose"
    },
    "gather": {
        "phi_name": "gather"
    },
    "gather_nd": {
        "phi_name": "gather_nd",
        "inputs": {
            "x": "X",
            "index": "Index"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "gather_tree": {
        "phi_name": "gather_tree",
        "inputs": {
            "ids": "Ids",
            "parents": "Parents"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "gaussian_random": {
        "phi_name": "gaussian",
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "shape": {
                "data_type": "int64_t",
                "tensor_name": "ShapeTensor",
                "tensors_name": "ShapeTensorList"
            }
        }
    },
    "gelu": {
        "phi_name": "gelu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "grad_add": {
        "phi_name": "grad_add"
    },
    "greater_equal": {
        "phi_name": "greater_equal",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "greater_than": {
        "phi_name": "greater_than",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "grid_sampler": {
        "phi_name": "grid_sample",
        "inputs": {
            "x": "X",
            "grid": "Grid"
        },
        "outputs": {
            "out": "Output"
        }
    },
    "group_norm": {
        "phi_name": "group_norm",
        "inputs": {
            "x": "X",
            "scale": "Scale",
            "bias": "Bias"
        },
        "outputs": {
            "y": "Y",
            "mean": "Mean",
            "variance": "Variance"
        }
    },
    "gru": {
        "phi_name": "gru"
    },
    "gumbel_softmax": {
        "phi_name": "gumbel_softmax",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "hard_shrink": {
        "phi_name": "hardshrink",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "hard_sigmoid": {
        "phi_name": "hardsigmoid",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "hard_swish": {
        "phi_name": "hardswish",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "brelu": {
        "phi_name": "hardtanh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "elementwise_heaviside": {
        "phi_name": "heaviside"
    },
    "histogram": {
        "phi_name": "histogram",
        "inputs": {
            "input": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "huber_loss": {
        "phi_name": "huber_loss",
        "inputs": {
            "input": "X",
            "label": "Y"
        },
        "outputs": {
            "out": "Out",
            "residual": "Residual"
        }
    },
    "imag": {
        "phi_name": "imag",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "index_add": {
        "phi_name": "index_add",
        "inputs": {
            "x": "X",
            "index": "Index",
            "add_value": "AddValue"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "index_sample": {
        "phi_name": "index_sample",
        "inputs": {
            "x": "X",
            "index": "Index"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "index_select": {
        "phi_name": "index_select",
        "inputs": {
            "x": "X",
            "index": "Index"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim"
        }
    },
    "inplace_abn": {
        "phi_name": "inplace_abn"
    },
    "instance_norm": {
        "phi_name": "instance_norm",
        "inputs": {
            "x": "X",
            "scale": "Scale",
            "bias": "Bias"
        },
        "outputs": {
            "y": "Y",
            "saved_mean": "SavedMean",
            "saved_variance": "SavedVariance"
        }
    },
    "inverse": {
        "phi_name": "inverse",
        "inputs": {
            "x": "Input"
        },
        "outputs": {
            "out": "Output"
        }
    },
    "is_empty": {
        "phi_name": "is_empty",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "isclose": {
        "phi_name": "isclose",
        "inputs": {
            "x": "Input",
            "y": "Other"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "rtol": {
                "data_type": "std::string",
                "tensor_name": "Rtol"
            },
            "atol": {
                "data_type": "std::string",
                "tensor_name": "Atol"
            }
        }
    },
    "isfinite_v2": {
        "phi_name": "isfinite",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "isinf_v2": {
        "phi_name": "isinf",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "isnan_v2": {
        "phi_name": "isnan",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "kldiv_loss": {
        "phi_name": "kldiv_loss",
        "inputs": {
            "x": "X",
            "label": "Target"
        },
        "outputs": {
            "out": "Loss"
        }
    },
    "kron": {
        "phi_name": "kron",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "kthvalue": {
        "phi_name": "kthvalue",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "indices": "Indices"
        }
    },
    "label_smooth": {
        "phi_name": "label_smooth",
        "inputs": {
            "label": "X",
            "prior_dist": "PriorDist"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "lamb_": {
        "phi_name": "lamb_",
        "inputs": {
            "param": "Param",
            "grad": "Grad",
            "learning_rate": "LearningRate",
            "moment1": "Moment1",
            "moment2": "Moment2",
            "beta1_pow": "Beta1Pow",
            "beta2_pow": "Beta2Pow",
            "master_param": "MasterParam",
            "skip_update": "SkipUpdate"
        },
        "outputs": {
            "param_out": "ParamOut",
            "moment1_out": "Moment1Out",
            "moment2_out": "Moment2Out",
            "beta1_pow_out": "Beta1PowOut",
            "beta2_pow_out": "Beta2PowOut",
            "master_param_outs": "MasterParamOut"
        }
    },
    "layer_norm": {
        "phi_name": "layer_norm",
        "inputs": {
            "x": "X",
            "scale": "Scale",
            "bias": "Bias"
        },
        "outputs": {
            "out": "Y",
            "mean": "Mean",
            "variance": "Variance"
        }
    },
    "leaky_relu": {
        "phi_name": "leaky_relu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "negative_slope": "alpha"
        }
    },
    "lerp": {
        "phi_name": "lerp",
        "inputs": {
            "x": "X",
            "y": "Y",
            "weight": "Weight"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "less_equal": {
        "phi_name": "less_equal",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "less_than": {
        "phi_name": "less_than",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "lgamma": {
        "phi_name": "lgamma",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "linear_interp_v2": {
        "phi_name": "linear_interp",
        "inputs": {
            "x": "X",
            "out_size": "OutSize",
            "size_tensor": "SizeTensor",
            "scale_tensor": "Scale"
        },
        "outputs": {
            "output": "Out"
        }
    },
    "linspace": {
        "phi_name": "linspace",
        "inputs": {
            "start": "Start",
            "stop": "Stop",
            "number": "Num"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "log": {
        "phi_name": "log",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "log10": {
        "phi_name": "log10",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "log1p": {
        "phi_name": "log1p",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "log2": {
        "phi_name": "log2",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "log_loss": {
        "phi_name": "log_loss",
        "inputs": {
            "input": "Predicted",
            "label": "Labels"
        },
        "outputs": {
            "out": "Loss"
        }
    },
    "log_softmax": {
        "phi_name": "log_softmax",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "logit": {
        "phi_name": "logit",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "logsigmoid": {
        "phi_name": "logsigmoid",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "lrn": {
        "phi_name": "lrn"
    },
    "lu_unpack": {
        "phi_name": "lu_unpack",
        "inputs": {
            "x": "X",
            "y": "Pivots"
        },
        "outputs": {
            "pmat": "Pmat",
            "l": "L",
            "u": "U"
        }
    },
    "margin_cross_entropy": {
        "phi_name": "margin_cross_entropy",
        "inputs": {
            "logits": "Logits",
            "label": "Label"
        },
        "outputs": {
            "softmax": "Softmax",
            "loss": "Loss"
        }
    },
    "masked_select": {
        "phi_name": "masked_select",
        "inputs": {
            "x": "X",
            "mask": "Mask"
        },
        "outputs": {
            "out": "Y"
        }
    },
    "matmul_v2": {
        "phi_name": "matmul",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "transpose_x": "trans_x",
            "transpose_y": "trans_y"
        }
    },
    "mul": {
        "phi_name": "matmul_with_flatten"
    },
    "matrix_power": {
        "phi_name": "matrix_power",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "reduce_max": {
        "phi_name": "max",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim",
            "keepdim": "keep_dim"
        },
        "int_array": {
            "axis": {
                "data_type": "int"
            }
        }
    },
    "elementwise_max": {
        "phi_name": "maximum"
    },
    "elementwise_min": {
        "phi_name": "maximum"
    },
    "maxout": {
        "phi_name": "maxout",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "reduce_mean": {
        "phi_name": "mean",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim",
            "keepdim": "keep_dim"
        }
    },
    "merge_selected_rows": {
        "phi_name": "merge_selected_rows",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "meshgrid": {
        "phi_name": "meshgrid",
        "inputs": {
            "inputs": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "mish": {
        "phi_name": "mish"
    },
    "mode": {
        "phi_name": "mode",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "indices": "Indices"
        }
    },
    "multi_dot": {
        "phi_name": "multi_dot",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "multinomial": {
        "phi_name": "multinomial",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "num_samples": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "multiplex": {
        "phi_name": "multiplex",
        "inputs": {
            "inputs": "X",
            "index": "Ids"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "elementwise_mul": {
        "phi_name": "multiply",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "mv": {
        "phi_name": "mv",
        "inputs": {
            "x": "X",
            "vec": "Vec"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "nanmedian": {
        "phi_name": "nanmedian",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "medians": "MedianIndex"
        },
        "int_array": {
            "axis": {
                "data_type": "int"
            }
        }
    },
    "nce": {
        "phi_name": "nce"
    },
    "nearest_interp_v2": {
        "phi_name": "nearest_interp",
        "inputs": {
            "x": "X",
            "out_size": "OutSize",
            "size_tensor": "SizeTensor",
            "scale_tensor": "Scale"
        },
        "outputs": {
            "output": "Out"
        }
    },
    "nll_loss": {
        "phi_name": "nll_loss",
        "inputs": {
            "input": "X",
            "label": "Label",
            "weight": "Weight"
        },
        "outputs": {
            "out": "Out",
            "total_weight": "Total_weight"
        }
    },
    "nms": {
        "phi_name": "nms",
        "inputs": {
            "x": "Boxes"
        },
        "outputs": {
            "out": "KeepBoxesIdxs"
        },
        "attrs": {
            "threshold": "iou_threshold"
        }
    },
    "where_index": {
        "phi_name": "nonzero",
        "inputs": {
            "condition": "Condition"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "not_equal": {
        "phi_name": "not_equal",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "size": {
        "phi_name": "numel",
        "inputs": {
            "x": "Input"
        },
        "outputs": {
            "size": "Out"
        }
    },
    "one_hot_v2": {
        "phi_name": "one_hot",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "depth": {
                "data_type": "int",
                "tensor_name": "depth_tensor"
            }
        }
    },
    "overlap_add": {
        "phi_name": "overlap_add",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "p_norm": {
        "phi_name": "p_norm",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "pad2d": {
        "phi_name": "pad2d"
    },
    "pad3d": {
        "phi_name": "pad3d"
    },
    "partial_sum": {
        "phi_name": "partial_sum"
    },
    "pixel_shuffle": {
        "phi_name": "pixel_shuffle",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "poisson": {
        "phi_name": "poisson",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "pool2d": {
        "phi_name": "pool2d"
    },
    "pool3d": {
        "phi_name": "pool3d"
    },
    "pow": {
        "phi_name": "pow",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "y": "factor"
        },
        "scalar": {
            "y": {
                "data_type": "float",
                "tensor_name": "FactorTensor"
            }
        }
    },
    "prelu": {
        "phi_name": "prelu",
        "inputs": {
            "x": "X",
            "alpha": "Alpha"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "reduce_prod": {
        "phi_name": "prod",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "dims": "dim",
            "keep_dim": "keep_dim"
        },
        "int_array": {
            "axis": {
                "data_type": "int"
            }
        }
    },
    "put_along_axis": {
        "phi_name": "put_along_axis",
        "inputs": {
            "arr": "Input",
            "indices": "Index",
            "values": "Value"
        },
        "outputs": {
            "out": "Result"
        },
        "attrs": {
            "axis": "Axis",
            "reduce": "Reduce"
        }
    },
    "qr": {
        "phi_name": "qr",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "q": "Q",
            "r": "R"
        }
    },
    "quantize_linear": {
        "phi_name": "quantize_linear"
    },
    "real": {
        "phi_name": "real",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "reciprocal": {
        "phi_name": "reciprocal",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "reduce_amax": {
        "phi_name": "reduce_amax"
    },
    "reduce_amin": {
        "phi_name": "reduce_amin"
    },
    "reduce_any": {
        "phi_name": "reduce_any"
    },
    "reduce_min": {
        "phi_name": "reduce_min"
    },
    "relu": {
        "phi_name": "relu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "relu6": {
        "phi_name": "relu6"
    },
    "elementwise_mod": {
        "phi_name": "remainder"
    },
    "renorm": {
        "phi_name": "renorm",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "reshape2": {
        "phi_name": "reshape",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "shape": {
                "data_type": "int",
                "tensor_name": "Shape",
                "tensors_name": "ShapeTensor"
            }
        }
    },
    "roll": {
        "phi_name": "roll",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "shifts": {
                "data_type": "int64_t",
                "tensor_name": "ShiftsTensor"
            }
        }
    },
    "round": {
        "phi_name": "round",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "rsqrt": {
        "phi_name": "rsqrt",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "scale": {
        "phi_name": "scale",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "scalar": {
            "scale": {
                "data_type": "float",
                "tensor_name": "ScaleTensor"
            }
        }
    },
    "scatter": {
        "phi_name": "scatter",
        "inputs": {
            "x": "X",
            "index": "Ids",
            "updates": "Updates"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "scatter_nd_add": {
        "phi_name": "scatter_nd_add",
        "inputs": {
            "x": "X",
            "index": "Index",
            "updates": "Updates"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "searchsorted": {
        "phi_name": "searchsorted",
        "inputs": {
            "sorted_sequence": "SortedSequence",
            "values": "Values"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "seed": {
        "phi_name": "seed"
    },
    "selu": {
        "phi_name": "selu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "graph_send_recv": {
        "phi_name": "send_u_recv",
        "inputs": {
            "x": "X",
            "src_index": "Src_index",
            "dst_index": "Dst_index"
        },
        "outputs": {
            "out": "Out",
            "dst_count": "Dst_count"
        },
        "int_array": {
            "out_size": {
                "data_type": "int64_t",
                "tensor_name": "Out_size"
            }
        }
    },
    "graph_send_ue_recv": {
        "phi_name": "send_ue_recv",
        "inputs": {
            "x": "X",
            "y": "Y",
            "src_index": "Src_index",
            "dst_index": "Dst_index"
        },
        "outputs": {
            "out": "Out",
            "dst_count": "Dst_count"
        },
        "int_array": {
            "out_size": {
                "data_type": "int64_t",
                "tensor_name": "Out_size"
            }
        }
    },
    "graph_send_uv": {
        "phi_name": "send_uv"
    },
    "sequence_softmax": {
        "phi_name": "sequence_softmax"
    },
    "sgd_": {
        "phi_name": "sgd_",
        "inputs": {
            "param": "Param",
            "learning_rate": "LearningRate",
            "grad": "Grad",
            "master_param": "MasterParam"
        },
        "outputs": {
            "param_out": "ParamOut",
            "master_param_out": "MasterParamOut"
        }
    },
    "shape": {
        "phi_name": "shape"
    },
    "shard_index": {
        "phi_name": "shard_index",
        "inputs": {
            "input": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "share_buffer": {
        "phi_name": "share_buffer",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "xout": "XOut"
        }
    },
    "shuffle_channel": {
        "phi_name": "shuffle_channel"
    },
    "sigmoid": {
        "phi_name": "sigmoid",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "sign": {
        "phi_name": "sign",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "silu": {
        "phi_name": "silu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "sin": {
        "phi_name": "sin",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "sinh": {
        "phi_name": "sinh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "slice": {
        "phi_name": "slice"
    },
    "slogdeterminant": {
        "phi_name": "slogdet",
        "inputs": {
            "x": "Input"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "softmax": {
        "phi_name": "softmax",
        "inputs": {
            "x": "X"
        }
    },
    "softplus": {
        "phi_name": "softplus",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "softshrink": {
        "phi_name": "softshrink",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "threshold": "lambda"
        }
    },
    "softsign": {
        "phi_name": "softsign",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "solve": {
        "phi_name": "solve",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "spectral_norm": {
        "phi_name": "spectral_norm",
        "inputs": {
            "weight": "Weight",
            "u": "U",
            "v": "V"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "sqrt": {
        "phi_name": "sqrt",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "square": {
        "phi_name": "square",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "squeeze2": {
        "phi_name": "squeeze",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "xshape": "XShape"
        },
        "attrs": {
            "axis": "axes"
        },
        "int_array": {
            "axis": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "stack": {
        "phi_name": "stack",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Y"
        }
    },
    "stanh": {
        "phi_name": "stanh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "elementwise_sub": {
        "phi_name": "subtract",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "reduce_sum": {
        "phi_name": "sum",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "attrs": {
            "axis": "dim",
            "keepdim": "keep_dim",
            "dtype": "out_dtype"
        },
        "int_array": {
            "axis": {
                "data_type": "int"
            }
        }
    },
    "svd": {
        "phi_name": "svd",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "u": "U",
            "s": "S",
            "vh": "VH"
        }
    },
    "swish": {
        "phi_name": "swish"
    },
    "sync_batch_norm": {
        "phi_name": "sync_batch_norm"
    },
    "take_along_axis": {
        "phi_name": "take_along_axis",
        "inputs": {
            "arr": "Input",
            "indices": "Index"
        },
        "outputs": {
            "out": "Result"
        },
        "attrs": {
            "axis": "Axis"
        }
    },
    "tan": {
        "phi_name": "tan",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "tanh": {
        "phi_name": "tanh",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "tanh_shrink": {
        "phi_name": "tanh_shrink",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "thresholded_relu": {
        "phi_name": "thresholded_relu",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "tile": {
        "phi_name": "tile",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out"
        },
        "int_array": {
            "repeat_times": {
                "data_type": "int",
                "tensor_name": "RepeatTimes",
                "tensors_name": "repeat_times_tensor"
            }
        }
    },
    "top_k_v2": {
        "phi_name": "topk",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "indices": "Indices"
        },
        "scalar": {
            "k": {
                "data_type": "int",
                "tensor_name": "K"
            }
        }
    },
    "trace": {
        "phi_name": "trace",
        "inputs": {
            "x": "Input"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "transpose2": {
        "phi_name": "transpose",
        "attrs": {
            "perm": "axis"
        }
    },
    "trilinear_interp_v2": {
        "phi_name": "trilinear_interp",
        "inputs": {
            "x": "X",
            "out_size": "OutSize",
            "size_tensor": "SizeTensor",
            "scale_tensor": "Scale"
        },
        "outputs": {
            "output": "Out"
        }
    },
    "trunc": {
        "phi_name": "trunc",
        "inputs": {
            "input": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "unbind": {
        "phi_name": "unbind",
        "inputs": {
            "input": "X"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "unfold": {
        "phi_name": "unfold",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Y"
        }
    },
    "unique_consecutive": {
        "phi_name": "unique_consecutive",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "index": "Index",
            "counts": "Counts"
        }
    },
    "unsqueeze2": {
        "phi_name": "unsqueeze",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Out",
            "xshape": "XShape"
        },
        "attrs": {
            "axis": "axes"
        },
        "int_array": {
            "axis": {
                "data_type": "int",
                "tensor_name": "AxesTensor",
                "tensors_name": "AxesTensorList"
            }
        }
    },
    "unstack": {
        "phi_name": "unstack",
        "inputs": {
            "x": "X"
        },
        "outputs": {
            "out": "Y"
        }
    },
    "update_loss_scaling_": {
        "phi_name": "update_loss_scaling_",
        "inputs": {
            "x": "X",
            "found_infinite": "FoundInfinite",
            "prev_loss_scaling": "PrevLossScaling",
            "in_good_steps": "InGoodSteps",
            "in_bad_steps": "InBadSteps"
        },
        "outputs": {
            "out": "Out",
            "loss_scaling": "LossScaling",
            "out_good_steps": "OutGoodSteps",
            "out_bad_steps": "OutBadSteps"
        },
        "scalar": {
            "stop_update": {
                "data_type": "bool",
                "tensor_name": "StopUpdate"
            }
        }
    },
    "viterbi_decode": {
        "phi_name": "viterbi_decode",
        "inputs": {
            "potentials": "Input",
            "transition_params": "Transition",
            "lengths": "Length"
        },
        "outputs": {
            "scores": "Scores",
            "path": "Path"
        }
    },
    "where": {
        "phi_name": "where",
        "inputs": {
            "condition": "Condition",
            "x": "X",
            "y": "Y"
        },
        "outputs": {
            "out": "Out"
        }
    },
    "while": {
        "phi_name": "while"
    },
    "sigmoid_cross_entropy_with_logits": {
        "phi_name": "sigmoid_cross_entropy_with_logits",
        "inputs": {
            "x": "X",
            "label": "Label"
        },
        "outputs": {
            "out": "Out"
        }
    }
}
op_info = {
    "accuracy": {
        "args": "Tensor x, Tensor indices, Tensor label",
        "output": "Tensor(accuracy), Tensor(correct), Tensor(total)"
    },
    "acos": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "acosh": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "addmm": {
        "args": "Tensor input, Tensor x, Tensor y, float beta=1.0, float alpha=1.0",
        "output": "Tensor"
    },
    "allclose": {
        "args": "Tensor x, Tensor y, Scalar rtol=\"1e-5\", Scalar atol=\"1e-8\", bool equal_nan=false",
        "output": "Tensor(out)"
    },
    "angle": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "argsort": {
        "args": "Tensor x, int axis=-1, bool descending=false",
        "output": "Tensor(out), Tensor(indices)"
    },
    "as_complex": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "as_real": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "asin": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "asinh": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "atan": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "atan2": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "atanh": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "auc": {
        "args": "Tensor x, Tensor label, Tensor stat_pos, Tensor stat_neg, Tensor ins_tag_weight, str curve = \"ROC\", int num_thresholds = (2 << 12) - 1, int slide_steps = 1",
        "output": "Tensor(auc), Tensor(stat_pos_out), Tensor(stat_neg_out)"
    },
    "bce_loss": {
        "args": "Tensor input, Tensor label",
        "output": "Tensor"
    },
    "bernoulli": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "bicubic_interp": {
        "args": "Tensor x, Tensor out_size, Tensor[] size_tensor, Tensor scale_tensor, str data_layout=\"NCHW\", int out_d=0, int out_h=0, int out_w=0, float[] scale={}, str interp_method=\"bilinear\", bool align_corners=true, int align_mode=1",
        "output": "Tensor(output)"
    },
    "bilinear_interp": {
        "args": "Tensor x, Tensor out_size, Tensor[] size_tensor, Tensor scale_tensor, str data_layout=\"NCHW\", int out_d=0, int out_h=0, int out_w=0, float[] scale={}, str interp_method=\"bilinear\", bool align_corners=true, int align_mode=1",
        "output": "Tensor(output)"
    },
    "bitwise_and": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "bitwise_not": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "bitwise_or": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "bitwise_xor": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "bmm": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "broadcast_tensors": {
        "args": "Tensor[] input",
        "output": "Tensor[]{input.size()}"
    },
    "ceil": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "celu": {
        "args": "Tensor x, float alpha = 1.0",
        "output": "Tensor(out)"
    },
    "cholesky": {
        "args": "Tensor x, bool upper=false",
        "output": "Tensor"
    },
    "cholesky_solve": {
        "args": "Tensor x, Tensor y, bool upper=false",
        "output": "Tensor"
    },
    "clip": {
        "args": "Tensor x, Scalar(float) min, Scalar(float) max",
        "output": "Tensor(out)"
    },
    "complex": {
        "args": "Tensor real, Tensor imag",
        "output": "Tensor"
    },
    "conj": {
        "args": "Tensor x",
        "output": "Tensor (out)"
    },
    "cos": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "cosh": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "crop": {
        "args": "Tensor x, IntArray shape = {}, IntArray offsets = {}",
        "output": "Tensor(out)"
    },
    "cross": {
        "args": "Tensor x, Tensor y, int axis = 9",
        "output": "Tensor"
    },
    "cumprod": {
        "args": "Tensor x,  int dim",
        "output": "Tensor(out)"
    },
    "det": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "diag": {
        "args": "Tensor x, int offset = 0, float padding_value = 0.0",
        "output": "Tensor"
    },
    "diag_embed": {
        "args": "Tensor input, int offset = 0, int dim1 = -2, int dim2 = -1",
        "output": "Tensor(out)"
    },
    "diagonal": {
        "args": "Tensor x, int offset = 0, int axis1 = 0, int axis2 = 1",
        "output": "Tensor"
    },
    "digamma": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "dirichlet": {
        "args": "Tensor alpha",
        "output": "Tensor(out)"
    },
    "dist": {
        "args": "Tensor x, Tensor y, float p = 2.0",
        "output": "Tensor"
    },
    "dot": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "eig": {
        "args": "Tensor x",
        "output": "Tensor(out_w), Tensor(out_v)"
    },
    "eigh": {
        "args": "Tensor x, str UPLO = \"L\"",
        "output": "Tensor(out_w), Tensor(out_v)"
    },
    "eigvals": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "elu": {
        "args": "Tensor x, float alpha = 1.0f",
        "output": "Tensor(out)"
    },
    "equal_all": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "erf": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "erfinv": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "exp": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "expm1": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "fft_c2c": {
        "args": "Tensor x, int64_t[] axes, str normalization, bool forward",
        "output": "Tensor"
    },
    "fft_c2r": {
        "args": "Tensor x, int64_t[] axes, str normalization, bool forward, int64_t last_dim_size=0L",
        "output": "Tensor"
    },
    "fft_r2c": {
        "args": "Tensor x, int64_t[] axes, str normalization, bool forward, bool onesided",
        "output": "Tensor"
    },
    "fill_diagonal": {
        "args": "Tensor x, float value=0, int offset=0, bool wrap=false",
        "output": "Tensor(out)"
    },
    "fill_diagonal_tensor": {
        "args": "Tensor x, Tensor y, int64_t offset = 0, int dim1 = 0, int dim2 = 1",
        "output": "Tensor(out)"
    },
    "flash_attn": {
        "args": "Tensor q, Tensor k, Tensor v, float dropout = 0.0, bool causal = false, bool return_softmax = false",
        "output": "Tensor(out), Tensor(softmax), Tensor(softmax_lse), Tensor(seed_offset)"
    },
    "flash_attn_unpadded": {
        "args": "Tensor q, Tensor k, Tensor v, Tensor cu_seqlens_q,  Tensor cu_seqlens_k, int64_t max_seqlen_q, int64_t max_seqlen_k, float scale, float dropout = 0.0, bool causal = false, bool return_softmax = false",
        "output": "Tensor(out), Tensor(softmax), Tensor(softmax_lse), Tensor(seed_offset)"
    },
    "flip": {
        "args": "Tensor x, int[] axis",
        "output": "Tensor (out)"
    },
    "floor": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "fold": {
        "args": "Tensor x, int[] output_sizes, int[] kernel_sizes,  int[] strides, int[] paddings, int[] dilations",
        "output": "Tensor(out)"
    },
    "frame": {
        "args": "Tensor x, int frame_length, int hop_length, int axis=-1",
        "output": "Tensor(out)"
    },
    "gather_nd": {
        "args": "Tensor x, Tensor index",
        "output": "Tensor"
    },
    "gather_tree": {
        "args": "Tensor ids, Tensor parents",
        "output": "Tensor(out)"
    },
    "gelu": {
        "args": "Tensor x,  bool approximate = false",
        "output": "Tensor(out)"
    },
    "grid_sample": {
        "args": "Tensor x, Tensor grid, str mode = \"bilinear\", str padding_mode = \"zeros\", bool align_corners = true",
        "output": "Tensor(out)"
    },
    "gumbel_softmax": {
        "args": "Tensor x, float temperature = 1.0, bool hard = false, int axis = -1",
        "output": "Tensor"
    },
    "hardshrink": {
        "args": "Tensor x, float threshold = 0.5",
        "output": "Tensor (out)"
    },
    "hardsigmoid": {
        "args": "Tensor x, float slope = 0.2, float offset = 0.5",
        "output": "Tensor (out)"
    },
    "hardtanh": {
        "args": "Tensor x, float t_min=0, float t_max=24",
        "output": "Tensor"
    },
    "histogram": {
        "args": "Tensor input, int64_t bins = 100, int min = 0, int max = 0",
        "output": "Tensor(out)"
    },
    "huber_loss": {
        "args": "Tensor input, Tensor label, float delta",
        "output": "Tensor(out), Tensor(residual)"
    },
    "imag": {
        "args": "Tensor x",
        "output": "Tensor (out)"
    },
    "index_add": {
        "args": "Tensor x, Tensor index,  Tensor add_value, int axis = 0",
        "output": "Tensor(out)"
    },
    "index_sample": {
        "args": "Tensor x, Tensor index",
        "output": "Tensor"
    },
    "index_select": {
        "args": "Tensor x, Tensor index, int axis = 0",
        "output": "Tensor(out)"
    },
    "inverse": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "is_empty": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "isclose": {
        "args": "Tensor x, Tensor y, Scalar rtol=\"1e-5\", Scalar atol=\"1e-8\",  bool equal_nan=false",
        "output": "Tensor(out)"
    },
    "isfinite": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "isinf": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "isnan": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "kldiv_loss": {
        "args": "Tensor x, Tensor label, str reduction = \"mean\"",
        "output": "Tensor(out)"
    },
    "kron": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "kthvalue": {
        "args": "Tensor x, int k = 1, int axis = -1, bool keepdim = false",
        "output": "Tensor(out), Tensor(indices)"
    },
    "label_smooth": {
        "args": "Tensor label, Tensor prior_dist, float epsilon = 0.0f",
        "output": "Tensor (out)"
    },
    "lamb_": {
        "args": "Tensor param, Tensor grad, Tensor learning_rate, Tensor moment1, Tensor moment2, Tensor beta1_pow, Tensor beta2_pow, Tensor master_param, Tensor skip_update, float weight_decay, float beta1=0.9, float beta2=0.999, float epsilon=1.0e-6f, bool multi_precision=false",
        "output": "Tensor(param_out), Tensor(moment1_out), Tensor(moment2_out), Tensor(beta1_pow_out), Tensor(beta2_pow_out), Tensor(master_param_outs)"
    },
    "leaky_relu": {
        "args": "Tensor x, float negative_slope = 0.02f",
        "output": "Tensor"
    },
    "lerp": {
        "args": "Tensor x, Tensor y, Tensor weight",
        "output": "Tensor(out)"
    },
    "lgamma": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "linear_interp": {
        "args": "Tensor x, Tensor out_size, Tensor[] size_tensor, Tensor scale_tensor, str data_layout=\"NCHW\", int out_d=0, int out_h=0, int out_w=0, float[] scale={}, str interp_method=\"bilinear\", bool align_corners=true, int align_mode=1",
        "output": "Tensor(output)"
    },
    "log": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "log10": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "log1p": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "log2": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "log_loss": {
        "args": "Tensor input, Tensor label, float epsilon",
        "output": "Tensor"
    },
    "log_softmax": {
        "args": "Tensor x, int axis = -1",
        "output": "Tensor(out)"
    },
    "logit": {
        "args": "Tensor x, float eps = 1e-6f",
        "output": "Tensor"
    },
    "logsigmoid": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "lu_unpack": {
        "args": "Tensor x, Tensor y, bool unpack_ludata = true, bool unpack_pivots = true",
        "output": "Tensor(pmat), Tensor(l), Tensor(u)"
    },
    "margin_cross_entropy": {
        "args": "Tensor logits, Tensor label, bool return_softmax = false, int ring_id = 0, int rank = 0, int nranks = 1, float margin1 = 1.0f, float margin2 = 0.5f, float margin3 = 0.0f, float scale = 64.0f",
        "output": "Tensor(softmax), Tensor(loss)"
    },
    "masked_select": {
        "args": "Tensor x, Tensor mask",
        "output": "Tensor (out)"
    },
    "matrix_power": {
        "args": "Tensor x, int n",
        "output": "Tensor"
    },
    "maxout": {
        "args": "Tensor x, int groups, int axis = 1",
        "output": "Tensor(out)"
    },
    "memory_efficient_attention": {
        "args": "Tensor query, Tensor key, Tensor value, Tensor bias, Tensor cu_seqlens_q, Tensor cu_seqlens_k, Tensor causal_diagonal, Tensor seqlen_k, Scalar max_seqlen_q, Scalar max_seqlen_k, bool causal, double dropout_p, float scale, bool is_test",
        "output": "Tensor(output), Tensor(logsumexp), Tensor(seed_and_offset)"
    },
    "merge_selected_rows": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "meshgrid": {
        "args": "Tensor[] inputs",
        "output": "Tensor[]{inputs.size()}"
    },
    "mode": {
        "args": "Tensor x,  int axis = -1,  bool keepdim = false",
        "output": "Tensor(out), Tensor(indices)"
    },
    "multi_dot": {
        "args": "Tensor[] x",
        "output": "Tensor"
    },
    "multinomial": {
        "args": "Tensor x, Scalar(int) num_samples = 1, bool replacement = false",
        "output": "Tensor(out)"
    },
    "multiplex": {
        "args": "Tensor[] inputs, Tensor index",
        "output": "Tensor"
    },
    "mv": {
        "args": "Tensor x, Tensor vec",
        "output": "Tensor"
    },
    "nanmedian": {
        "args": "Tensor x, IntArray axis = {}, bool keepdim = true",
        "output": "Tensor(out), Tensor(medians)"
    },
    "nearest_interp": {
        "args": "Tensor x, Tensor out_size, Tensor[] size_tensor, Tensor scale_tensor, str data_layout=\"NCHW\", int out_d=0, int out_h=0, int out_w=0, float[] scale={}, str interp_method=\"bilinear\", bool align_corners=true, int align_mode=1",
        "output": "Tensor(output)"
    },
    "nll_loss": {
        "args": "Tensor input, Tensor label, Tensor weight, int64_t ignore_index = -100, str reduction = \"mean\"",
        "output": "Tensor(out), Tensor(total_weight)"
    },
    "nms": {
        "args": "Tensor x, float threshold = 1.0f",
        "output": "Tensor(out)"
    },
    "nonzero": {
        "args": "Tensor condition",
        "output": "Tensor(out)"
    },
    "npu_identity": {
        "args": "Tensor x, int format = -1",
        "output": "Tensor"
    },
    "numel": {
        "args": "Tensor x",
        "output": "Tensor(size)"
    },
    "overlap_add": {
        "args": "Tensor x, int hop_length, int axis=-1",
        "output": "Tensor"
    },
    "p_norm": {
        "args": "Tensor x,  float porder=2,  int axis=-1,  float epsilon=1.0e-12f,  bool keepdim=false,  bool asvector=false",
        "output": "Tensor(out)"
    },
    "pixel_shuffle": {
        "args": "Tensor x, int upscale_factor=1, str data_format=\"NCHW\"",
        "output": "Tensor"
    },
    "poisson": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "pow": {
        "args": "Tensor x, Scalar y=1.0f",
        "output": "Tensor(out)"
    },
    "prelu": {
        "args": "Tensor x, Tensor alpha, str data_format=\"NCHW\", str mode=\"all\"",
        "output": "Tensor(out)"
    },
    "put_along_axis": {
        "args": "Tensor arr, Tensor indices, Tensor values, int axis, str reduce = \"assign\"",
        "output": "Tensor(out)"
    },
    "qr": {
        "args": "Tensor x, str mode = \"reduced\"",
        "output": "Tensor(q), Tensor(r)"
    },
    "real": {
        "args": "Tensor x",
        "output": "Tensor (out)"
    },
    "reciprocal": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "relu": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "renorm": {
        "args": "Tensor x, float p, int axis, float max_norm",
        "output": "Tensor"
    },
    "roll": {
        "args": "Tensor x, IntArray shifts={}, int64_t[] axis={}",
        "output": "Tensor(out)"
    },
    "round": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "rsqrt": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "scale": {
        "args": "Tensor x, Scalar scale=1.0, float bias=0.0, bool bias_after_scale=true",
        "output": "Tensor(out)"
    },
    "scatter": {
        "args": "Tensor x, Tensor index, Tensor updates, bool overwrite=true",
        "output": "Tensor(out)"
    },
    "scatter_nd_add": {
        "args": "Tensor x, Tensor index, Tensor updates",
        "output": "Tensor"
    },
    "searchsorted": {
        "args": "Tensor sorted_sequence, Tensor values, bool out_int32 = false, bool right = false",
        "output": "Tensor(out)"
    },
    "selu": {
        "args": "Tensor x, float scale=1.0507009873554804934193349852946, float alpha=1.6732632423543772848170429916717",
        "output": "Tensor"
    },
    "send_u_recv": {
        "args": "Tensor x, Tensor src_index, Tensor dst_index, str reduce_op = \"SUM\", IntArray out_size = {0}",
        "output": "Tensor(out), Tensor(dst_count)"
    },
    "send_ue_recv": {
        "args": "Tensor x, Tensor y, Tensor src_index, Tensor dst_index, str message_op=\"ADD\", str reduce_op=\"SUM\", IntArray out_size={0}",
        "output": "Tensor(out), Tensor(dst_count)"
    },
    "send_uv": {
        "args": "Tensor x, Tensor y, Tensor src_index, Tensor dst_index, str message_op = \"ADD\"",
        "output": "Tensor(out)"
    },
    "sgd_": {
        "args": "Tensor param, Tensor learning_rate, Tensor grad, Tensor master_param, bool multi_precision=false",
        "output": "Tensor(param_out), Tensor(master_param_out)"
    },
    "shard_index": {
        "args": "Tensor input, int index_num, int nshards, int shard_id, int ignore_value=-1",
        "output": "Tensor(out)"
    },
    "sigmoid": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "sigmoid_cross_entropy_with_logits": {
        "args": "Tensor x, Tensor label, bool normalize=false, int ignore_index=-100",
        "output": "Tensor"
    },
    "sign": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "silu": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "sin": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "sinh": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "slogdet": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "softplus": {
        "args": "Tensor x, float beta = 1.0, float threshold = 20.0f",
        "output": "Tensor"
    },
    "softshrink": {
        "args": "Tensor x, float threshold = 0.5",
        "output": "Tensor"
    },
    "softsign": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "solve": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "spectral_norm": {
        "args": "Tensor weight, Tensor u, Tensor v, int dim = 0, int power_iters = 1, float eps = 1e-12f",
        "output": "Tensor"
    },
    "sqrt": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "square": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "squeeze": {
        "args": "Tensor x, IntArray axis={}",
        "output": "Tensor(out), Tensor(xshape)"
    },
    "stack": {
        "args": "Tensor[] x, int axis = 0",
        "output": "Tensor (out)"
    },
    "stanh": {
        "args": "Tensor x, float scale_a=0.67f, float scale_b=1.7159f",
        "output": "Tensor(out)"
    },
    "svd": {
        "args": "Tensor x, bool full_matrices = false",
        "output": "Tensor(u), Tensor(s), Tensor(vh)"
    },
    "take_along_axis": {
        "args": "Tensor arr, Tensor indices, int axis",
        "output": "Tensor"
    },
    "tan": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "tanh": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "tanh_shrink": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "thresholded_relu": {
        "args": "Tensor x, float threshold = 1.0",
        "output": "Tensor"
    },
    "topk": {
        "args": "Tensor x, Scalar(int) k = 1, int axis = -1, bool largest = true, bool sorted = true",
        "output": "Tensor(out), Tensor(indices)"
    },
    "trace": {
        "args": "Tensor x, int offset = 0, int axis1 = 0, int axis2 = 1",
        "output": "Tensor"
    },
    "trilinear_interp": {
        "args": "Tensor x, Tensor out_size, Tensor[] size_tensor, Tensor scale_tensor, str data_layout=\"NCHW\", int out_d=0, int out_h=0, int out_w=0, float[] scale={}, str interp_method=\"bilinear\", bool align_corners=true, int align_mode=1",
        "output": "Tensor(output)"
    },
    "trunc": {
        "args": "Tensor input",
        "output": "Tensor"
    },
    "unbind": {
        "args": "Tensor input, int axis = 0",
        "output": "Tensor[] {axis<0 ? input.dims()[input.dims().size()+axis]:input.dims()[axis]}"
    },
    "unfold": {
        "args": "Tensor x, int[] kernel_sizes, int[] strides, int[] paddings, int[] dilations",
        "output": "Tensor(out)"
    },
    "unique_consecutive": {
        "args": "Tensor x, bool return_inverse = false, bool return_counts = false, int[] axis = {}, int dtype = 5",
        "output": "Tensor(out), Tensor(index), Tensor(counts)"
    },
    "unsqueeze": {
        "args": "Tensor x, IntArray axis = {}",
        "output": "Tensor(out), Tensor(xshape)"
    },
    "unstack": {
        "args": "Tensor x, int axis=0, int num=0",
        "output": "Tensor[](out){num}"
    },
    "update_loss_scaling_": {
        "args": "Tensor[] x, Tensor found_infinite, Tensor prev_loss_scaling, Tensor in_good_steps, Tensor in_bad_steps, int incr_every_n_steps, int decr_every_n_nan_or_inf, float incr_ratio, float decr_ratio, Scalar stop_update=false",
        "output": "Tensor[](out){x.size()}, Tensor(loss_scaling), Tensor(out_good_steps), Tensor(out_bad_steps)"
    },
    "viterbi_decode": {
        "args": "Tensor potentials, Tensor transition_params, Tensor lengths, bool include_bos_eos_tag = true",
        "output": "Tensor(scores), Tensor(path)"
    },
    "warprnnt": {
        "args": "Tensor input, Tensor label, Tensor input_lengths, Tensor label_lengths, int blank = 0, float fastemit_lambda = 0.0",
        "output": "Tensor(loss), Tensor(warprnntgrad)"
    },
    "where": {
        "args": "Tensor condition, Tensor x, Tensor y",
        "output": "Tensor"
    },
    "abs": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "adadelta_": {
        "args": "Tensor param, Tensor grad, Tensor avg_squared_grad, Tensor avg_squared_update, Tensor master_param, float rho, float epsilon, bool multi_precision",
        "output": "Tensor(param_out), Tensor(moment_out), Tensor(inf_norm_out), Tensor(master_param_out)"
    },
    "adagrad_": {
        "args": "Tensor param, Tensor grad, Tensor moment, Tensor learning_rate, Tensor master_param, float epsilon, bool multi_precision",
        "output": "Tensor(param_out), Tensor(moment_out), Tensor(master_param_out)"
    },
    "adam_": {
        "args": "Tensor param, Tensor grad, Tensor learning_rate, Tensor moment1, Tensor moment2, Tensor beta1_pow, Tensor beta2_pow, Tensor master_param, Tensor skip_update, Scalar beta1, Scalar beta2, Scalar epsilon, bool lazy_mode, int64_t min_row_size_to_use_multithread, bool multi_precision, bool use_global_beta_pow",
        "output": "Tensor(param_out), Tensor(moment1_out), Tensor(moment2_out), Tensor(beta1_pow_out), Tensor(beta2_pow_out), Tensor(master_param_outs)"
    },
    "adamax_": {
        "args": "Tensor param, Tensor grad, Tensor learning_rate, Tensor moment, Tensor inf_norm, Tensor beta1_pow, Tensor master_param, float beta1, float beta2, float epsilon, bool multi_precision",
        "output": "Tensor(param_out), Tensor(avg_squared_grad_out), Tensor(avg_squared_update_out), Tensor(master_param_outs)"
    },
    "adamw_": {
        "args": "Tensor param, Tensor grad, Tensor learning_rate, Tensor moment1, Tensor moment2, Tensor beta1_pow, Tensor beta2_pow, Tensor master_param, Tensor skip_update, Scalar beta1, Scalar beta2, Scalar epsilon, float lr_ratio, float coeff, bool with_decay, bool lazy_mode, int64_t min_row_size_to_use_multithread, bool multi_precision, bool use_global_beta_pow",
        "output": "Tensor(param_out), Tensor(moment1_out), Tensor(moment2_out), Tensor(beta1_pow_out), Tensor(beta2_pow_out), Tensor(master_param_outs)"
    },
    "add": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "add_n": {
        "args": "Tensor[] inputs",
        "output": "Tensor"
    },
    "affine_grid": {
        "args": "Tensor input, IntArray outputShape, bool align_corners=true",
        "output": "Tensor"
    },
    "all": {
        "args": "Tensor x, int64_t[] axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "amax": {
        "args": "Tensor x, int64_t[] axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "amin": {
        "args": "Tensor x, int64_t[] axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "any": {
        "args": "Tensor x, int64_t[] axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "arange": {
        "args": "Tensor start, Tensor end, Tensor step, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "argmax": {
        "args": "Tensor x, Scalar axis, bool keepdims, bool flatten, int dtype",
        "output": "Tensor(out)"
    },
    "argmin": {
        "args": "Tensor x, Scalar axis, bool keepdims, bool flatten, int dtype",
        "output": "Tensor(out)"
    },
    "assign": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "assign_out_": {
        "args": "Tensor x, Tensor output",
        "output": "Tensor(out)"
    },
    "assign_value_": {
        "args": "Tensor output, int[] shape, DataType dtype, Scalar[] values, Place place = {}",
        "output": "Tensor(out)"
    },
    "average_accumulates_": {
        "args": "Tensor param, Tensor in_sum_1, Tensor in_sum_2, Tensor in_sum_3, Tensor in_num_accumulates, Tensor in_old_num_accumulates, Tensor in_num_updates, float average_window, int64_t max_average_window, int64_t min_average_window",
        "output": "Tensor(out_sum_1), Tensor(out_sum_2), Tensor(out_sum_3), Tensor(out_num_accumulates), Tensor(out_old_num_accumulates), Tensor(out_num_updates)"
    },
    "batch_norm": {
        "args": "Tensor x, Tensor mean, Tensor variance, Tensor scale, Tensor bias, bool is_test, float momentum, float epsilon, str data_layout, bool use_global_stats, bool trainable_statistics",
        "output": "Tensor(out), Tensor(mean_out), Tensor(variance_out), Tensor(saved_mean), Tensor(saved_variance), Tensor(reserve_space)"
    },
    "bilinear_tensor_product": {
        "args": "Tensor x, Tensor y, Tensor weight, Tensor bias",
        "output": "Tensor"
    },
    "bincount": {
        "args": "Tensor x, Tensor weights, Scalar(int) minlength = 0",
        "output": "Tensor(out)"
    },
    "box_coder": {
        "args": "Tensor prior_box, Tensor prior_box_var, Tensor target_box, str code_type, bool box_normalized, int axis, float[] variance",
        "output": "Tensor(output_box)"
    },
    "cast": {
        "args": "Tensor x, DataType dtype",
        "output": "Tensor"
    },
    "channel_shuffle": {
        "args": "Tensor x, int groups, str data_format=\"NCHW\"",
        "output": "Tensor(out)"
    },
    "check_finite_and_unscale_": {
        "args": "Tensor[] x, Tensor scale, Tensor input_found_infinite",
        "output": "Tensor[](out){x.size()}, Tensor(output_found_infinite)"
    },
    "class_center_sample": {
        "args": "Tensor label, int num_classes, int num_samples, int ring_id, int rank, int nranks, bool fix_seed, int seed",
        "output": "Tensor(remapped_label), Tensor(sampled_local_class_center)"
    },
    "clip_by_norm": {
        "args": "Tensor x, float max_norm",
        "output": "Tensor(out)"
    },
    "coalesce_tensor": {
        "args": "Tensor[] input, DataType dtype, bool copy_data = false, bool set_constant = false, bool persist_output = false, float constant = 0.0, bool use_align = true, int align_size = -1, int size_of_dtype = -1, int64_t[] concated_shapes = {}, int64_t[] concated_ranks = {}",
        "output": "Tensor[](output){input.size()}, Tensor(fused_output)"
    },
    "concat": {
        "args": "Tensor[] x, Scalar(int64_t) axis",
        "output": "Tensor"
    },
    "conv2d": {
        "args": "Tensor input, Tensor filter, int[] strides, int[] paddings, str padding_algorithm, int[] dilations, int groups, str data_format",
        "output": "Tensor"
    },
    "conv2d_transpose": {
        "args": "Tensor x, Tensor filter, int[] strides, int[] paddings, int[] output_padding, IntArray output_size, str padding_algorithm, int groups, int[] dilations, str data_format",
        "output": "Tensor(out)"
    },
    "conv3d": {
        "args": "Tensor input, Tensor filter, int[] strides, int[] paddings, str padding_algorithm, int groups, int[] dilations, str data_format",
        "output": "Tensor"
    },
    "conv3d_transpose": {
        "args": "Tensor x, Tensor filter, int[] strides, int[] paddings, int[] output_padding, int[] output_size, str padding_algorithm, int groups, int[] dilations, str data_format",
        "output": "Tensor(out)"
    },
    "copy_to": {
        "args": "Tensor x, Place place, bool blocking",
        "output": "Tensor(out)"
    },
    "cross_entropy_with_softmax": {
        "args": "Tensor input, Tensor label, bool soft_label, bool use_softmax, bool numeric_stable_mode, int ignore_index, int axis",
        "output": "Tensor(softmax), Tensor(loss)"
    },
    "cumsum": {
        "args": "Tensor x, Scalar axis, bool flatten, bool exclusive, bool reverse",
        "output": "Tensor(out)"
    },
    "decode_jpeg": {
        "args": "Tensor x, str mode, Place place",
        "output": "Tensor(out)"
    },
    "deformable_conv": {
        "args": "Tensor x, Tensor offset, Tensor filter, Tensor mask, int[] strides, int[] paddings, int[] dilations, int deformable_groups, int groups, int im2col_step",
        "output": "Tensor(out)"
    },
    "depthwise_conv2d": {
        "args": "Tensor x, Tensor filter, int[] strides, int[] paddings, str padding_algorithm, int groups, int[] dilations, str data_format",
        "output": "Tensor(out)"
    },
    "depthwise_conv2d_transpose": {
        "args": "Tensor x, Tensor filter, int[] strides, int[] paddings, int[] output_padding, IntArray output_size, str padding_algorithm, int groups, int[] dilations, str data_format",
        "output": "Tensor(out)"
    },
    "distribute_fpn_proposals": {
        "args": "Tensor fpn_rois, Tensor rois_num, int min_level, int max_level, int refer_level, int refer_scale, bool pixel_offset",
        "output": "Tensor[](multi_fpn_rois){max_level - min_level + 1}, Tensor[](multi_level_rois_num){max_level - min_level + 1}, Tensor(restore_index)"
    },
    "divide": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "dropout": {
        "args": "Tensor x, Tensor seed_tensor, Scalar p, bool is_test, str mode, int seed, bool fix_seed",
        "output": "Tensor(out), Tensor(mask)"
    },
    "edit_distance": {
        "args": "Tensor hyps, Tensor refs, Tensor hypslength, Tensor refslength, bool normalized = false",
        "output": "Tensor(sequencenum), Tensor(out)"
    },
    "eigvalsh": {
        "args": "Tensor x, str uplo, bool is_test",
        "output": "Tensor(eigenvalues), Tensor(eigenvectors)"
    },
    "einsum": {
        "args": "Tensor[] x, str equation",
        "output": "Tensor, Tensor[]{x.size()}, Tensor[]{x.size()}"
    },
    "elementwise_pow": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "embedding": {
        "args": "Tensor x, Tensor weight, int64_t padding_idx=-1, bool sparse=false",
        "output": "Tensor"
    },
    "empty": {
        "args": "IntArray shape, DataType dtype=DataType::FLOAT32, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "empty_like": {
        "args": "Tensor x, DataType dtype = DataType::UNDEFINED, Place place = {}",
        "output": "Tensor(out)"
    },
    "equal": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "expand": {
        "args": "Tensor x, IntArray shape",
        "output": "Tensor"
    },
    "expand_as": {
        "args": "Tensor x, Tensor y, int[] target_shape",
        "output": "Tensor(out)"
    },
    "exponential_": {
        "args": "Tensor x, float lam",
        "output": "Tensor(out)"
    },
    "eye": {
        "args": "Scalar num_rows, Scalar num_columns, DataType dtype=DataType::FLOAT32, Place place={}",
        "output": "Tensor(out)"
    },
    "fill": {
        "args": "Tensor x, Scalar value",
        "output": "Tensor(out)"
    },
    "flatten": {
        "args": "Tensor x, int start_axis, int stop_axis",
        "output": "Tensor(out), Tensor(xshape)"
    },
    "floor_divide": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "fmax": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "fmin": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "frobenius_norm": {
        "args": "Tensor x, int64_t[] axis,  bool keep_dim,  bool reduce_all",
        "output": "Tensor(out)"
    },
    "full": {
        "args": "IntArray shape, Scalar value, DataType dtype=DataType::FLOAT32, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "full_": {
        "args": "Tensor output, IntArray shape, Scalar value, DataType dtype=DataType::FLOAT32, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "full_batch_size_like": {
        "args": "Tensor input, int[] shape, DataType dtype, Scalar value, int input_dim_idx, int output_dim_idx, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "full_like": {
        "args": "Tensor x, Scalar value, DataType dtype = DataType::UNDEFINED, Place place = {}",
        "output": "Tensor(out)"
    },
    "fused_adam_": {
        "args": "Tensor[] params, Tensor[] grads, Tensor learning_rate, Tensor[] moments1, Tensor[] moments2, Tensor[] beta1_pows, Tensor[] beta2_pows, Tensor[] master_params, Tensor skip_update, Scalar beta1, Scalar beta2, Scalar epsilon, int chunk_size, float weight_decay, bool use_adamw, bool multi_precision, bool use_global_beta_pow",
        "output": "Tensor[](params_out){params.size()}, Tensor[](moments1_out){params.size()}, Tensor[](moments2_out){params.size()}, Tensor[](beta1_pows_out){params.size()}, Tensor[](beta2_pows_out){params.size()}, Tensor[](master_params_out){params.size()}"
    },
    "gather": {
        "args": "Tensor x, Tensor index, Scalar(int) axis=0",
        "output": "Tensor(out)"
    },
    "gaussian": {
        "args": "IntArray shape, float mean, float std, int seed, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "generate_proposals": {
        "args": "Tensor scores, Tensor bbox_deltas, Tensor im_shape, Tensor anchors, Tensor variances, int pre_nms_top_n, int post_nms_top_n, float nms_thresh, float min_size, float eta, bool pixel_offset=true",
        "output": "Tensor(rpn_rois), Tensor(rpn_roi_probs), Tensor(rpn_rois_num)"
    },
    "greater_equal": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "greater_than": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "group_norm": {
        "args": "Tensor x, Tensor scale, Tensor bias, float epsilon, int groups, str data_layout",
        "output": "Tensor(y), Tensor(mean), Tensor(variance)"
    },
    "hardswish": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "heaviside": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "hsigmoid_loss": {
        "args": "Tensor x, Tensor label, Tensor w, Tensor bias, Tensor path, Tensor code, int num_classes, bool remote_prefetch, bool is_sparse",
        "output": "Tensor(out), Tensor(pre_out), Tensor(w_out)"
    },
    "increment": {
        "args": "Tensor x, float value = 1.0",
        "output": "Tensor(out)"
    },
    "instance_norm": {
        "args": "Tensor x, Tensor scale, Tensor bias, float epsilon",
        "output": "Tensor(y), Tensor(saved_mean), Tensor(saved_variance)"
    },
    "layer_norm": {
        "args": "Tensor x, Tensor scale, Tensor bias, float epsilon, int begin_norm_axis",
        "output": "Tensor(out), Tensor(mean), Tensor(variance)"
    },
    "less_equal": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "less_than": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "linspace": {
        "args": "Tensor start, Tensor stop, Tensor number, DataType dtype, Place place",
        "output": "Tensor(out)"
    },
    "logcumsumexp": {
        "args": "Tensor x, int axis, bool flatten, bool exclusive, bool reverse",
        "output": "Tensor(out)"
    },
    "logical_and": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "logical_not": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "logical_or": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "logical_xor": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "logspace": {
        "args": "Tensor start, Tensor stop, Tensor num, Tensor base, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "logsumexp": {
        "args": "Tensor x, int64_t[] axis,  bool keepdim,  bool reduce_all",
        "output": "Tensor(out)"
    },
    "lstsq": {
        "args": "Tensor x, Tensor y, Scalar rcond, str driver",
        "output": "Tensor(solution), Tensor(residuals), Tensor(rank), Tensor(singular_values)"
    },
    "lu": {
        "args": "Tensor x, bool pivot",
        "output": "Tensor(out), Tensor(pivots), Tensor(infos)"
    },
    "matmul": {
        "args": "Tensor x, Tensor y, bool transpose_x = false, bool transpose_y = false",
        "output": "Tensor"
    },
    "matrix_nms": {
        "args": "Tensor bboxes, Tensor scores, float score_threshold, int nms_top_k, int keep_top_k, float post_threshold=0., bool use_gaussian = false, float gaussian_sigma = 2.0, int background_label = 0, bool normalized = true",
        "output": "Tensor(out), Tensor(index), Tensor(roisnum)"
    },
    "matrix_rank": {
        "args": "Tensor x, float tol, bool hermitian=false, bool use_default_tol=true",
        "output": "Tensor(out)"
    },
    "matrix_rank_tol": {
        "args": "Tensor x, Tensor atol_tensor, bool use_default_tol=true, bool hermitian=false",
        "output": "Tensor(out)"
    },
    "max": {
        "args": "Tensor x, IntArray axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "max_pool2d_with_index": {
        "args": "Tensor x, int[] kernel_size, int[] strides, int[] paddings, bool global_pooling, bool adaptive",
        "output": "Tensor(out), Tensor(mask)"
    },
    "max_pool3d_with_index": {
        "args": "Tensor x, int[] kernel_size, int[] strides, int[] paddings, bool global_pooling, bool adaptive",
        "output": "Tensor(out), Tensor(mask)"
    },
    "maximum": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "mean": {
        "args": "Tensor x, IntArray axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "mean_all": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "merged_adam_": {
        "args": "Tensor[] param, Tensor[] grad, Tensor[] learning_rate, Tensor[] moment1, Tensor[] moment2, Tensor[] beta1_pow, Tensor[] beta2_pow, Tensor[] master_param, Scalar beta1, Scalar beta2, Scalar epsilon, bool multi_precision, bool use_global_beta_pow",
        "output": "Tensor[](param_out){param.size()}, Tensor[](moment1_out){param.size()}, Tensor[](moment2_out){param.size()}, Tensor[](beta1_pow_out){param.size()}, Tensor[](beta2_pow_out){param.size()}, Tensor[](master_param_out){param.size()}"
    },
    "merged_momentum_": {
        "args": "Tensor[] param, Tensor[] grad, Tensor[] velocity, Tensor[] learning_rate, Tensor[] master_param, float mu, bool use_nesterov = false, str[] regularization_method = {}, float[] regularization_coeff = {}, bool multi_precision = false, float rescale_grad = 1.0f",
        "output": "Tensor[](param_out){param.size()}, Tensor[](velocity_out){param.size()}, Tensor[](master_param_out){param.size()}"
    },
    "min": {
        "args": "Tensor x, IntArray axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "minimum": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "mish": {
        "args": "Tensor x, float lambda",
        "output": "Tensor"
    },
    "momentum_": {
        "args": "Tensor param, Tensor grad, Tensor velocity, Tensor learning_rate, Tensor master_param, float mu, bool use_nesterov = false, str regularization_method = \"\", float regularization_coeff = 0.0, bool multi_precision = false, float rescale_grad = 1.0f",
        "output": "Tensor(param_out), Tensor(velocity_out), Tensor(master_param_out)"
    },
    "multiclass_nms3": {
        "args": "Tensor bboxes, Tensor scores, Tensor rois_num, float score_threshold, int nms_top_k, int keep_top_k, float nms_threshold=0.3, bool normalized=true, float nms_eta=1.0, int background_label=0",
        "output": "Tensor(out), Tensor(index), Tensor(nms_rois_num)"
    },
    "multiply": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "norm": {
        "args": "Tensor x, int axis, float epsilon, bool is_test",
        "output": "Tensor(out), Tensor(norm)"
    },
    "not_equal": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "one_hot": {
        "args": "Tensor x, Scalar(int) num_classes",
        "output": "Tensor(out)"
    },
    "ones": {
        "args": "IntArray shape, DataType dtype=DataType::FLOAT32, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "ones_like": {
        "args": "Tensor x, DataType dtype=DataType::UNDEFINED, Place place={}",
        "output": "Tensor(out)"
    },
    "pad": {
        "args": "Tensor x, int[] paddings, Scalar pad_value",
        "output": "Tensor"
    },
    "pad3d": {
        "args": "Tensor x, IntArray paddings, str mode,  float pad_value, str data_format",
        "output": "Tensor(out)"
    },
    "pool2d": {
        "args": "Tensor x, IntArray kernel_size, int[] strides, int[] paddings, bool ceil_mode, bool exclusive, str data_format, str pooling_type, bool global_pooling, bool adaptive, str padding_algorithm",
        "output": "Tensor(out)"
    },
    "pool3d": {
        "args": "Tensor x, int[] kernel_size, int[] strides, int[] paddings, bool ceil_mode, bool exclusive, str data_format, str pooling_type, bool global_pooling, bool adaptive, str padding_algorithm",
        "output": "Tensor(out)"
    },
    "prior_box": {
        "args": "Tensor input, Tensor image, float[] min_sizes, float[] aspect_ratios, float[] variances, float[] max_sizes = {}, bool flip=true, bool clip=true, float step_w=0.0, float step_h=0.0, float offset=0.5, bool min_max_aspect_ratios_order=false",
        "output": "Tensor(out), Tensor(var)"
    },
    "prod": {
        "args": "Tensor x, IntArray dims, bool keep_dim, bool reduce_all",
        "output": "Tensor"
    },
    "psroi_pool": {
        "args": "Tensor x, Tensor boxes, Tensor boxes_num, int pooled_height, int pooled_width, int output_channels, float spatial_scale",
        "output": "Tensor"
    },
    "randint": {
        "args": "int low, int high, IntArray shape, DataType dtype=DataType::INT64, Place place={}",
        "output": "Tensor(out)"
    },
    "randperm": {
        "args": "int n, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "relu6": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "remainder": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor (out)"
    },
    "repeat_interleave": {
        "args": "Tensor x, int repeats, int axis",
        "output": "Tensor(out)"
    },
    "repeat_interleave_with_tensor_index": {
        "args": "Tensor x, Tensor repeats, int axis",
        "output": "Tensor(out)"
    },
    "reshape": {
        "args": "Tensor x, IntArray shape",
        "output": "Tensor(out), Tensor(xshape)"
    },
    "reverse": {
        "args": "Tensor x, IntArray axis",
        "output": "Tensor"
    },
    "rmsprop_": {
        "args": "Tensor param, Tensor mean_square, Tensor grad, Tensor moment, Tensor learning_rate, Tensor mean_grad, Tensor master_param, float epsilon, float decay, float momentum, bool centered, bool multi_precision",
        "output": "Tensor(param_out), Tensor(moment_out), Tensor(mean_square_out), Tensor(mean_grad_out), Tensor(master_param_out)"
    },
    "rnn": {
        "args": "Tensor x, Tensor[] pre_state, Tensor[] weight_list, Tensor sequence_length, Tensor dropout_state_in, float dropout_prob=0.0, bool is_bidirec=false, int input_size=10, int hidden_size=100, int num_layers=1, str mode=\"RNN_TANH\", int seed=0, bool is_test=false",
        "output": "Tensor(out), Tensor(dropout_state_out), Tensor[](state){pre_state.size()}, Tensor(reserve)"
    },
    "roi_align": {
        "args": "Tensor x, Tensor boxes, Tensor boxes_num, int pooled_height, int pooled_width, float spatial_scale, int sampling_ratio, bool aligned",
        "output": "Tensor"
    },
    "roi_pool": {
        "args": "Tensor x, Tensor boxes, Tensor boxes_num, int pooled_height, int pooled_width, float spatial_scale",
        "output": "Tensor(out), Tensor(arg_max)"
    },
    "rrelu": {
        "args": "Tensor x, float lower, float upper, bool is_test",
        "output": "Tensor(out), Tensor(noise)"
    },
    "segment_pool": {
        "args": "Tensor x, Tensor segment_ids, str pooltype",
        "output": "Tensor(out), Tensor(summed_ids)"
    },
    "shape": {
        "args": "Tensor input",
        "output": "Tensor(out)"
    },
    "slice": {
        "args": "Tensor input, int64_t[] axes, IntArray starts, IntArray ends, int64_t[] infer_flags, int64_t[] decrease_axis",
        "output": "Tensor"
    },
    "softmax": {
        "args": "Tensor x, int axis",
        "output": "Tensor(out)"
    },
    "split": {
        "args": "Tensor x, IntArray sections, Scalar(int) axis",
        "output": "Tensor[]{sections.size()}"
    },
    "split_with_num": {
        "args": "Tensor x, int num, Scalar(int) axis",
        "output": "Tensor[]{num}"
    },
    "squared_l2_norm": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "strided_slice": {
        "args": "Tensor x, int[] axes, IntArray starts, IntArray ends, IntArray strides",
        "output": "Tensor"
    },
    "subtract": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "sum": {
        "args": "Tensor x, IntArray axis={}, DataType dtype=DataType::UNDEFINED, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "swish": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "sync_batch_norm_": {
        "args": "Tensor x, Tensor mean, Tensor variance, Tensor scale, Tensor bias, bool is_test, float momentum, float epsilon, str data_layout, bool use_global_stats, bool trainable_statistics",
        "output": "Tensor(out), Tensor(mean_out), Tensor(variance_out), Tensor(saved_mean), Tensor(saved_variance), Tensor(reserve_space)"
    },
    "temporal_shift": {
        "args": "Tensor x, int seg_num, float shift_ratio, str data_format_str",
        "output": "Tensor"
    },
    "tile": {
        "args": "Tensor x, IntArray repeat_times = {}",
        "output": "Tensor(out)"
    },
    "transpose": {
        "args": "Tensor x, int[] perm",
        "output": "Tensor"
    },
    "triangular_solve": {
        "args": "Tensor x, Tensor y, bool upper, bool transpose, bool unitriangular",
        "output": "Tensor"
    },
    "tril": {
        "args": "Tensor x, int diagonal",
        "output": "Tensor(out)"
    },
    "tril_indices": {
        "args": "int rows, int cols, int offset, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "triu": {
        "args": "Tensor x, int diagonal",
        "output": "Tensor(out)"
    },
    "triu_indices": {
        "args": "int row, int col, int offset, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "truncated_gaussian_random": {
        "args": "int[] shape, float mean, float std, int seed, DataType dtype=DataType::FLOAT32, Place place={}",
        "output": "Tensor(out)"
    },
    "uniform": {
        "args": "IntArray shape,  DataType dtype,  Scalar min,  Scalar max,  int seed, Place place={}",
        "output": "Tensor(out)"
    },
    "uniform_inplace": {
        "args": "Tensor x, float min, float max, int seed, int diag_num, int diag_step, float diag_val",
        "output": "Tensor(out)"
    },
    "unique": {
        "args": "Tensor x, bool return_index, bool return_inverse, bool return_counts, int[] axis, DataType dtype=DataType::INT64",
        "output": "Tensor(out), Tensor(indices), Tensor(inverse), Tensor(counts)"
    },
    "unpool": {
        "args": "Tensor x, Tensor indices, int[] ksize, int[] strides, int[] padding, IntArray output_size, str data_format",
        "output": "Tensor(out)"
    },
    "unpool3d": {
        "args": "Tensor x, Tensor indices, int[] ksize, int[] strides, int[] padding, int[] output_size, str data_format",
        "output": "Tensor(out)"
    },
    "warpctc": {
        "args": "Tensor logits, Tensor label, Tensor logits_length, Tensor labels_length, int blank, bool norm_by_times",
        "output": "Tensor(loss), Tensor(warpctcgrad)"
    },
    "yolo_box": {
        "args": "Tensor x, Tensor img_size, int[] anchors, int class_num, float conf_thresh, int downsample_ratio, bool clip_bbox, float scale_x_y=1.0, bool iou_aware=false, float iou_aware_factor=0.5",
        "output": "Tensor(boxes), Tensor(scores)"
    },
    "yolo_loss": {
        "args": "Tensor x, Tensor gt_box, Tensor gt_label, Tensor gt_score, int[] anchors, int[] anchor_mask, int class_num, float ignore_thresh, int downsample_ratio, bool use_label_smooth=true, float scale_x_y=1.0",
        "output": "Tensor(loss), Tensor(objectness_mask), Tensor(gt_match_mask)"
    },
    "zeros": {
        "args": "IntArray shape, DataType dtype=DataType::FLOAT32, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "zeros_like": {
        "args": "Tensor x, DataType dtype=DataType::UNDEFINED, Place place = {}",
        "output": "Tensor(out)"
    }
}

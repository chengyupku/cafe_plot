llama_providers = ["bs=1, s=4096", "bs=1, s=1", "bs=32, s=1"]

llama_times_data_h100 = [
    ("TileSight", [12.7812, 0.6968, 2.1996]),
    ("TileSight-Base", [13.3982, 0.6968, 3.7434]),
    ("TileSight-Wint2Aint8", [7.0692, 0.2248, 1.034, ]),
    ("Ladder", [16.4578, 0.7663, 5.1107, ]),
    ("PyTorch Inductor", [24.02202, 0.934182, 4.597709, ]),
    ("TensorRT", [38.0424, 1.70215, 6.16897, ]),
]


llama_times_data_mi210 = [
    ("TileSight", [48.44, 1.58, 18.4, ]),
    ("TileSight-Base", [64.0782, 2.1932, 24.4725, ]),
    ("TileSight-Wint2Aint8", [45.326748, 0.410433279, 6.4948, ]),
    ("Ladder", [64.0782, 2.1932, 24.4725, ]),
    ("PyTorch Inductor", [87.194453, 3.679322, 22.143532, ]),
]

qwen_providers = ["bs=1, s=4096", "bs=1, s=1", "bs=32, s=1"]

qwen_times_data_h100 = [
    ("TileSight", [21.3741, 1.1746, 5.5692, ]),
    ("TileSight-Base", [21.9911, 1.1823, 7.113, ]),
    ("TileSight-Wint2Aint8", [10.9661, 0.5616, 3.721, ]),
    ("Ladder", [36.1909, 1.3409, 9.9483, ]),
    ("PyTorch Inductor", [38.926106, 1.325664, 5.687258, ]),
    ("TensorRT", [31.3832, 1.24668, 5.1949, ]),
]


qwen_times_data_mi210 = [
    ("TileSight", [72.5, 2.1, 21.6, ]),
    ("TileSight-Base", [90.68135023, 2.98208107, 29.153931, ]),
    ("TileSight-Wint2Aint8", [68.76572, 0.725347, 8.53607, ]),
    ("Ladder", [90.68135023, 2.98208107, 29.153931, ]),
    ("PyTorch Inductor", [144.765498, 5.013629, 25.916665, ]),
]


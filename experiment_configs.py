"""
experiment_configs.py

Unified configuration file for SDN experiments.
Used by:
- main.py (Single receiver experiments)
- research/generate_paper_data.py (Data generation)
- research/paper_figures_spatial.py (Visualization)
"""

import numpy as np
from rir_calculators import calculate_pra_rir, calculate_rimpy_rir, calculate_sdn_rir, calculate_sdn_rir_fast, calculate_ho_sdn_rir

# =============================================================================
# ROOM SETUP
# =============================================================================

room_aes = {
    'display_name': 'AES Room',
    'width': 9, 'depth': 7, 'height': 4,
    'source x': 4.5, 'source y': 3.5, 'source z': 2,
    'mic x': 2, 'mic y': 2, 'mic z': 1.5,
    'absorption': 0.2,
}

room_aes_rx00 = {
    'display_name': 'AES Room',
    'width': 9, 'depth': 7, 'height': 4,
    'source x': 4.5, 'source y': 3.5, 'source z': 2,
    'mic x': 0.5, 'mic y': 0.5, 'mic z': 1.5,
    'absorption': 0.2,
}

room_waspaa = {
    'display_name': 'WASPAA Room',
    'width': 6, 'depth': 7, 'height': 4,
    'source x': 3.6, 'source y': 5.3, 'source z': 1.3,
    'mic x': 1.2, 'mic y': 1.8, 'mic z': 2.4,
    'absorption': 0.1,
}

room_journal = {
    'display_name': 'Journal Room',
    'width': 3.2, 'depth': 4, 'height': 2.7,
    'source x': 2, 'source y': 3., 'source z': 2,
    'mic x': 1, 'mic y': 1, 'mic z': 1.5,
    'absorption': 0.1,
}

# Default Active Room
active_room = room_aes

if active_room['display_name'] == 'Journal Room':
    duration = 1.2
elif active_room['display_name'] == 'WASPAA Room':
    duration = 1.8
elif active_room['display_name'] == 'AES Room':
    duration = 1
else:
    assert  False, "Unknown room selected!"

Fs = 44100

# =============================================================================
# METHOD FLAGS - Toggle these to enable/disable methods
# =============================================================================

# Reference methods - ISM
PLOT_ISM_with_pra = False
PLOT_ISM_with_pra_rand10 = False
PLOT_ISM_rimPy_pos = False
PLOT_ISM_rimPy_pos_rand10 = False
PLOT_ISM_rimPy_neg = False
PLOT_ISM_rimPy_neg_rand10 = False  #

# tests
wall1, wall2, wall3, wall4, wall5, wall6 = False, False, False, False, False, False
t1111, t111, t11 = False, False, False
t1,t2,t3,t4,t5 = False, False, False, False, False
fast1, fast2 = False, False
testx = False

# SDN Tests
RUN_SDN_Test0 = False
RUN_SDN_Test1 = False  # c=1 original
RUN_SDN_Test2 = False
RUN_SDN_Test3 = False
RUN_SDN_Test2_998 = True # Test2.998
RUN_SDN_Test4_71 = False # Test2.998
RUN_SDN_Test4 = False
RUN_SDN_Test5 = False
RUN_SDN_Test6 = False
RUN_SDN_Test7 = False

RUN_SDN_Test_micX = False
RUN_SDN_Test2_mic = False
RUN_SDN_Test3_mic = False
RUN_SDN_Test4_mic = False
RUN_SDN_Test5_mic = False

SDN_SW_v2_kk000 = False
SDN_SW_v2_kkk00 = False
# Per-source optimized c values (from optimization)
RUN_SDN_c_center = False     # c=4.71 for center source
RUN_SDN_c_lower_left = False # c=2.06 for lower-left source
RUN_SDN_c_top_middle = False # c=2.99 for top-middle source
RUN_SDN_c_upper_right = False # c=1.24 for upper-right source

RUN_SDN_Test1b = False
RUN_SDN_Test1c = False

RUN_SDN_Test3r = False
RUN_SDN_Test4r = False
RUN_SDN_Test5r = False
RUN_SDN_Test6r = False
RUN_SDN_Test7r = False

# Fast SDN Tests
fast1 = False
fast2 = False
RUN_SDN_Test1_FAST = False
RUN_SDN_Test2_FAST = False
RUN_SDN_Test2_998_FAST = False
RUN_SDN_Test3_FAST = False
RUN_SDN_Test4_FAST = False
RUN_SDN_Test5_FAST = False
RUN_SDN_Test6_FAST = False
RUN_SDN_Test7_FAST = False

# SDN Tests without attenuation
RUN_SDN_Test1_noatt =   False
RUN_SDN_Test2_noatt =   False

# HO-SDN Tests (Reference Leny)
RUN_HO_N1 = False
RUN_HO_N2 = False
RUN_HO_N2g = False
RUN_HO_N3 = False
RUN_HO_N3g = False

# HO-SDN Tests (My Implementation)
RUN_MY_HO_SDN_n1 = False
RUN_MY_HO_SDN_n2 = False
RUN_MY_HO_SDN_n3 = False
RUN_MY_HO_SDN_n2_swc5 = False
RUN_MY_HO_SDN_n2_swc3 = False
RUN_MY_HO_SDN_n3_swc3 = False
RUN_MY_HO_SDN_n1noatt = False
RUN_MY_HO_SDN_n2noatt = False

# =============================================================================
# METHOD CONFIGURATIONS
# =============================================================================

# ISM Methods
ism_methods = {
    'ISM-pra': {
        'enabled': PLOT_ISM_with_pra,
        'info': 'pra 100',
        'function': calculate_pra_rir,
        'calculator': 'pra',
        'params': {'max_order': 100, 'use_rand_ism': False}
    },
    'ISM-pra-rand10': {
        'enabled': PLOT_ISM_with_pra_rand10,
        'info': 'pra 100 + 10cm randomness',
        'function': calculate_pra_rir,
        'calculator': 'pra',
        'params': {'max_order': 100, 'use_rand_ism': True, 'max_rand_disp': 0.1}
    },
    'RIMPY-pos': {
        'enabled': PLOT_ISM_rimPy_pos,
        'info': 'Positive Reflection',
        'function': calculate_rimpy_rir,
        'calculator': 'rimpy',
        'params': {'reflection_sign': 1, 'tw_fractional_delay_length': 0}
    },
    'RIMPY-pos-rand10': {
        'enabled': PLOT_ISM_rimPy_pos_rand10,
        'info': 'Positive Reflection + 10cm randomness',
        'function': calculate_rimpy_rir,
        'calculator': 'rimpy',
        'params': {'reflection_sign': 1, 'tw_fractional_delay_length': 0, 'randDist': 0.1}
    },
    'RIMPY-neg': {
        'enabled': PLOT_ISM_rimPy_neg,
        'info': 'Negative Reflection',
        'function': calculate_rimpy_rir,
        'calculator': 'rimpy',
        'params': {'reflection_sign': -1, 'tw_fractional_delay_length': 0}
    },
    'RIMPY-neg10': {
        'enabled': PLOT_ISM_rimPy_neg_rand10,
        'info': 'Negative Reflection + 10cm Randomness',
        'function': calculate_rimpy_rir,
        'calculator': 'rimpy',
        'params': {'reflection_sign': -1, 'tw_fractional_delay_length': 0, 'randDist': 0.1}
    }
}

# SDN Tests
sdn_tests = {

    'SDN-Test2_mic': {
                    'enabled': RUN_SDN_Test2_mic,
                    'info': "c2",
                    'calculator': 'sdn',
                    'flags': {
                    'specular_mic_pickup': True,
                     'mic_weighting': 2.0,
                    },
                    'label': "SDN Test 2 mic pickıp"
                },

    'SDN-Test3_mic': {
                    'enabled': RUN_SDN_Test3_mic,
                    'info': "c3",
                    'calculator': 'sdn',
                    'flags': {
                    'specular_mic_pickup': True,
                     'mic_weighting': 3.0,
                    },
                    'label': "SDN Test 3 mic pickıp"
                },

    'SDN-Test4_mic': {
                'enabled': RUN_SDN_Test4_mic,
                'info': "c4",
                'calculator': 'sdn',
                'flags': {
                'specular_mic_pickup': True,
                 'mic_weighting': 4.0,
                },
                'label': "SDN Test 4 mic pickıp"
            },

    'SDN-Test5_mic': {
            'enabled': RUN_SDN_Test5_mic,
            'info': "m5",
            'calculator': 'sdn',
            'flags': {
            'specular_mic_pickup': True,
             'mic_weighting': 5,
            },
            'label': ""
        },

    # Example: Using collection_vector (5-element vector) for mic pickup weighting
    'SDN-Test_mic_vector': {
        'enabled': RUN_SDN_Test_micX,
        'info': "X col + mic kök5",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'injection_vector': [np.sqrt(5),0,0,0,0],
            'specular_mic_pickup': True,
            'collection_vector': [0, 0, np.sqrt(5), 0, 0],  # [dominant, non-dom1, non-dom2, non-dom3, non-dom4]
        },
        'label': "SDN X"
    },


    'SDN-SW_v2_kk000': {
        'enabled': SDN_SW_v2_kk000,
        'info': "[2.5, 2.5, 0,0,0]",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'injection_vector': [2.5, 2.5, 0,0,0],
        },
        'label': "SDN"
    },

    'SDN-SW_v2_kkk00': {
            'enabled': SDN_SW_v2_kkk00,
            'info': "[5/3,5/3,5/3,0,0]",
            'calculator': 'sdn',
            'flags': {
                'specular_source_injection': True,
                'injection_vector': [5/3,5/3,5/3,0,0],
            },
            'label': "SDN"
        },

    'SDN-fast1': {
        'enabled': fast1, 'use_fast_method': True,
        'info': "fast [4.35,6.05,2.85,4.02,1.68,2.17]",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'injection_c_vector':[4.35,6.05,2.85,4.02,1.68,2.17]
        }, 
        'label': "SDN"
    },
    'SDN-fast2': {
        'enabled': fast2, 'use_fast_method': True,
        'info': "optimized c-vector",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'injection_c_vector':[4.79,6.20,1.00,7.00,1.00,2.09]
        }, 
        'label': "SDN"
    },

    'SDN-fast2_998': {
            'enabled': RUN_SDN_Test2_998_FAST,
        'use_fast_method': True,
            'info': "optimized c-vector",
            'calculator': 'sdn',
            'flags': {
                'specular_source_injection': True,
                'source_weighting': 2.998,
            },
            'label': "SDN"
        },

    'SDN-fast4_71': {
                'enabled': RUN_SDN_Test4_71,
                'info': "test 4.71",
                'calculator': 'sdn',
                'flags': {
                    'specular_source_injection': True,
                    'source_weighting': 4.71,
                },
                'label': "SDN"
            },

'SDN-Test2.998':  {
        'enabled': RUN_SDN_Test2_998,
        'info': "c 2.998 (global optimized)",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 2.998,
        }, 
        'label': ""
    },
    
    # Per-source optimized c values
    'SDN-c_center': {
        'enabled': RUN_SDN_c_center,
        'info': "c=4.71 (center source optimized)",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 4.71,
        },
        'label': "SDN"
    },
    'SDN-c_lower_left': {
        'enabled': RUN_SDN_c_lower_left,
        'info': "c=2.06 (lower-left source optimized)",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 2.06,
        },
        'label': "SDN"
    },
    'SDN-c_top_middle': {
        'enabled': RUN_SDN_c_top_middle,
        'info': "c=2.99 (top-middle source optimized)",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 2.99,
        },
        'label': "SDN"
    },
    'SDN-c_upper_right': {
        'enabled': RUN_SDN_c_upper_right,
        'info': "c=1.24 (upper-right source optimized)",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 1.24,
        },
        'label': "SDN"
    },
    
    'SDN-Test0': {
        'enabled': RUN_SDN_Test0,
        'info': "c0",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 0,
        },
        'label': "SDN"
    },
    'SDN-Test1': {
        'enabled': RUN_SDN_Test1,
        'info': "c1 original",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 1,
        },
        'label': "SDN"
    },
    'SDN-Test2': {
        'enabled': RUN_SDN_Test2,
        'info': "c2",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 2,
        },
        'label': "SDN"
    },
    'SDN-Test3': {
        'enabled': RUN_SDN_Test3,
        'info': "c3",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 3,
        },
        'label': "SDN"
    },
    'SDN-Test4': {
        'enabled': RUN_SDN_Test4,
        'info': "c4",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 4,
        },
        'label': "SDN"
    },
    'SDN-Test5': {
        'enabled': RUN_SDN_Test5,
        'info': "c5",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 5,
        },
        'label': "SDN Test 5"
    },
    'SDN-Test6': {
        'enabled': RUN_SDN_Test6,
        'info': "c6",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 6,
        },
        'label': "SDN Test 6"
    },
    'SDN-Test7': {
        'enabled': RUN_SDN_Test7,
        'info': "c7",
        'calculator': 'sdn',
        'flags': {
            'specular_source_injection': True,
            'source_weighting': 7,
        },
        'label': "SDN Test 7"
    },

    'Test1b': {'enabled': RUN_SDN_Test1b,
               'info': "c1 orj SN",
               'flags': {
                   'specular_source_injection': True,
                   'source_weighting': 1,
                   'specular_node_pressure': True,
               },
               'label': "SDN"
               },

    'Test1c': {'enabled': RUN_SDN_Test1c,
               'info': "c1 orj SP MAT",
               'flags': {
                   'specular_source_injection': True,
                   'source_weighting': 1,
                   'specular_scattering': True,
                   'specular_increase_coef': 0.02,
               },
               'label': "SDN"
               },
               
    # My HO-SDN Implementation (placed in sdn_tests as per user request)
    'TestHO_N1': {
        'enabled': RUN_MY_HO_SDN_n1,
        'info': "HO-SDN order 1",
        'calculator': 'ho_sdn',
        'order': 1,
        'source_signal': 'dirac',
        'label': "HO-SDN N=1"
    },
    'TestHO_N2': {
        'enabled': RUN_MY_HO_SDN_n2,
        'info': "HO-SDN order 2",
        'calculator': 'ho_sdn',
        'order': 2,
        'source_signal': 'dirac',
        'label': "HO-SDN N=2"
    },
    'TestHO_N3': {
        'enabled': RUN_MY_HO_SDN_n3,
        'info': "HO-SDN order 3",
        'calculator': 'ho_sdn',
        'order': 3,
        'source_signal': 'dirac',
        'label': "HO-SDN N=3"
    },
    'TestHO_N2_swc5': {
        'enabled': RUN_MY_HO_SDN_n2_swc5,
        'info': "sw-c5-ho-N2",
        'calculator': 'ho_sdn',
        'order': 2,
        'source_weighting': 5,
        'source_signal': 'dirac',
        'label': "SW-c5-HO-N2"
    },
    'TestHO_N2_swc3': {
        'enabled': RUN_MY_HO_SDN_n2_swc3,
        'info': "sw-c3-ho-N2",
        'calculator': 'ho_sdn',
        'order': 2,
        'source_weighting': 3,
        'source_signal': 'dirac',
        'label': "SW-c3-HO-N2"
    },
    'TestHO_N3_swc3': {
        'enabled': RUN_MY_HO_SDN_n3_swc3,
        'info': "sw-c3-ho-N3",
        'calculator': 'ho_sdn',
        'order': 3,
        'source_weighting': 3,
        'source_signal': 'dirac',
        'label': "SW-c3-HO-N3"
    },
}

# HO-SDN Tests (Reference Implementation)
ho_sdn_tests = {
    'N1': {
        'enabled': RUN_HO_N1,
        'info': "Dirac",
        'source_signal': 'dirac',
        'order': 1,
        'label': "HO-SDN N1"
    },
    'N2': {
        'enabled': RUN_HO_N2,
        'info': "Dirac",
        'source_signal': 'dirac',
        'order': 2,
        'label': "HO-SDN N2"
    },
    'N2g': {
        'enabled': RUN_HO_N2g,
        'info': "Gaussian",
        'source_signal': 'gaussian',
        'order': 2,
        'label': "HO-SDN N2"
    },
    'N3': {
        'enabled': RUN_HO_N3,
        'info': "Dirac",
        'source_signal': 'dirac',
        'order': 3,
        'label': "HO-SDN N3"
    },
    'N3g': {
        'enabled': RUN_HO_N3g,
        'info': "Gaussian",
        'source_signal': 'gaussian',
        'order': 3,
        'label': "HO-SDN N3"
    }
}

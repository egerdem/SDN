import numpy as np
import geometry
import matplotlib.pyplot as plt
import pickle 

from analysis import plot_room as pp
from analysis import path_tracker
from analysis.sdn_path_calculator import SDNCalculator, ISMCalculator, PathCalculator
from analysis import frequency as ff
from analysis import EchoDensity as ned
from analysis import analysis as an

from rir_calculators import calculate_pra_rir, calculate_rimpy_rir, calculate_sdn_rir
from rir_calculators import calculate_ho_sdn_rir, rir_normalisation

""" Method flags """
PLOT_SDN_BASE = False

t1,t2,t3,t4,t5 = False, False, False, False, False
RUN_SDN_Test_2 = False
RUN_SDN_Test_3 = False
RUN_SDN_Test0 = False
RUN_SDN_Test1 = True
RUN_SDN_Test2 = False
RUN_SDN_Test3 = False
RUN_SDN_Test4 = False
RUN_SDN_Test5 = True
RUN_SDN_Test6 = False
RUN_SDN_Test7 = False

RUN_SDN_Test3r = False
RUN_SDN_Test4r = False
RUN_SDN_Test5r = False
RUN_SDN_Test6r = False
RUN_SDN_Test7r = False

RUN_SDN_Test_2_noatt =  False
RUN_SDN_Test_3_noatt =  False
RUN_SDN_Test1_noatt =   False
RUN_SDN_Test2_noatt =   False
RUN_SDN_Test3_noatt =   False
RUN_SDN_Test4_noatt =   False
RUN_SDN_Test5_noatt =   False
RUN_SDN_Test6_noatt =   False

RUN_SDN_Test_2b = False
RUN_SDN_Test_3b = False
RUN_SDN_Test1b = False
RUN_SDN_Test2b = False
RUN_SDN_Test3b = False
RUN_SDN_Test4b = False
RUN_SDN_Test5b = False
RUN_SDN_Test6b = False

RUN_SDN_Test_3c = False
RUN_SDN_Test_2c = False
RUN_SDN_Test1c = False
RUN_SDN_Test3c = False
RUN_SDN_Test4c = False
RUN_SDN_Test5c = False
RUN_SDN_Test6c = False

RUN_HO_N1 = False
RUN_HO_N2 = False
RUN_HO_N2g = False
RUN_HO_N3 = False
RUN_HO_N3g = False

RUN_MY_HO_SDN_n1 = False # My new test flag for HO-SDN
RUN_MY_HO_SDN_n2_swc5 = False
RUN_MY_HO_SDN_n2_swc3 = False
RUN_MY_HO_SDN_n3_swc3 = False

RUN_MY_HO_SDN_n1noatt = False
RUN_MY_HO_SDN_n2noatt = False # My new test flag for HO-SDN no att

PLOT_TREBLE = False

PLOT_ISM_with_pra = False
PLOT_ISM_with_pra_rand10 = True
PLOT_ISM_rimPy_pos = False  # rimPy ISM with positive reflection
PLOT_ISM_rimPy_pos_rand10 = False
PLOT_ISM_rimPy_neg = False  # rimPy ISM with negative reflection
PLOT_ISM_rimPy_neg_rand10 = False
pra_order = 100

PICKLE_LOAD_RIRS = False # Load RIRs from pickle file
# file_name = "rirs_c_cSN_cSPMAT_rimneg_ho3ho2.pkl"
# file_name = "rirs_c_cSN_cSPMAT_rimneg_ho3ho2.pkl"
# file_name = "rirs_c_cSN.pkl"
file_name = "pra_n2swc3_n2swc5_n3_swc3_hoN2_hoN3_c5_c1.pkl"  #

""" Visualization flags """
PLOT_ROOM = False  # 3D Room Visualisation
PLOT_ISM_PATHS = False  # Visualize example ISM paths
ISM_SDN_PATH_DIFF_TABLE = False  # Run path analysis (ISM vs SDN comparison, invalid paths, visualization)
PLOT_REFLECTION_LINES = True  # Plot vertical lines at reflection arrival times
SAVE_audio = False

""" Analysis flags """
PLOT_EDC = False
PLOT_NED = False  # Plot Normalized Echo Density
PLOT_lsd = False  # Plot LSD
PLOT_FREQ = False  # Frequency response plot
UNIFIED_PLOTS = True  # Flag to switch between unified and separated plots
normalize_to_first_impulse = True  # Set this to True if you want to normalize to first impulse

Print_RIR_comparison_metrics = True
interactive_rirs = True  # Set to True to enable interactive RIR comparison
pulse_analysis = "upto_4"
plot_smoothed_rirs = False

""" Room Setup """

# ho-waspaa paper room
room_waspaa = {
    'display_name': 'WASPAA Room',
    'width': 6, 'depth': 7, 'height': 4,
    'source x': 3.6, 'source y': 5.3, 'source z': 1.3,
    'mic x': 1.2, 'mic y': 1.8, 'mic z': 2.4,
    'absorption': 0.1,
}

room_aes = {
    'display_name': 'AES Room',
    'width': 9, 'depth': 7, 'height': 4,
    'source x': 4.5, 'source y': 3.5, 'source z': 2,
    'mic x': 2, 'mic y': 2, 'mic z': 1.5,
    'absorption': 0.2,
}

room_aes_outliar = {
    'display_name': 'AES Room',
    'width': 9, 'depth': 7, 'height': 4,
    'source x': 8, 'source y': 6, 'source z': 2,
    'mic x': 0.5, 'mic y': 0.5, 'mic z': 1.5,
    'absorption': 0.2,
}

room_journal = {
    'display_name': 'Journal Room',
    'width': 3.2, 'depth': 4, 'height': 2.7,
    'source x': 2, 'source y': 3., 'source z': 2,
    'mic x': 2.83, 'mic y': 3.00, 'mic z': 1.5,
    'absorption': 0.1,
}

# room_journal_random = {
#         'display_name': 'Journal Room',
#         'width': 3.2, 'depth': 4, 'height': 2.7,
#         'source x': 1.5, 'source y': 2.4, 'source z': 2,
#         'mic x': 1.3, 'mic y': 3, 'mic z': 1.4,
#         'absorption': 0.1,
#     }

# room_parameters = room_aes  # Choose the room
# room_parameters = room_waspaa  # Choose the room
# room_parameters = room_journal
room_parameters = room_aes_outliar

# Parameters
duration = 1  # seconds
duration_in_ms = 1000 * duration  # Convert to milliseconds
Fs = 44100
num_samples = int(Fs * duration)
rirs = {}
default_rirs = set()  # Track which RIRs should be black
rirs_analysis = {}
# Assign source signals
impulse_dirac = geometry.Source.generate_signal('dirac', num_samples)
impulse_gaussian = geometry.Source.generate_signal('gaussian', num_samples)

# print room name and duration of the experiment
print(f"\n=== {room_parameters['display_name']} ===")
print(f"Duration: {duration} seconds")

room = geometry.Room(room_parameters['width'], room_parameters['depth'], room_parameters['height'])
room.set_microphone(room_parameters['mic x'], room_parameters['mic y'], room_parameters['mic z'])
room.set_source(room_parameters['source x'], room_parameters['source y'], room_parameters['source z'],
                signal="will be replaced", Fs=Fs)

room_dim = np.array([room_parameters['width'], room_parameters['depth'], room_parameters['height']])

# Setup signal
room.source.signal = impulse_dirac['signal']
# room.source.signal = impulse_gaussian['signal']

# Calculate reflection coefficient - will be overwritten for each method
room_parameters['reflection'] = np.sqrt(1 - room_parameters['absorption'])
room.wallAttenuation = [room_parameters['reflection']] * 6

# ISM and rimPy Test Configurations
ism_methods = {

    'ISM-pra': {
        'enabled': PLOT_ISM_with_pra,
        'info': 'pra 100',
        'function': calculate_pra_rir,
        'params': {
            'max_order': 100,
            'use_rand_ism': False,
        }
    },

    'ISM-pra-rand10': {
        'enabled': PLOT_ISM_with_pra_rand10,
        'info': 'pra 100',
        'function': calculate_pra_rir,
        'params': {
            'max_order': 100,
            'use_rand_ism': True,
            'max_rand_disp': 0.1
        }
    },
    'rimPy-pos': {
        'enabled': PLOT_ISM_rimPy_pos,
        'info': 'Positive Reflection',
        'function': calculate_rimpy_rir,
        'params': {
            'reflection_sign': 1,
            'tw_fractional_delay_length': 0
        }
    },

    'rimPy-pos-rand10': {
            'enabled': PLOT_ISM_rimPy_pos_rand10,
            'info': 'Positive Reflection + 10cm randomness',
            'function': calculate_rimpy_rir,
            'params': {
                'reflection_sign': 1,
                'tw_fractional_delay_length': 0,
                'randDist': 0.1
            }
        },

    'rimPy-neg': {
        'enabled': PLOT_ISM_rimPy_neg,
        'info': 'Negative Reflection',
        'function': calculate_rimpy_rir,
        'params': {
            'reflection_sign': -1,
            'tw_fractional_delay_length': 0
        }
    },
    'rimPy-neg-rand10': {
        'enabled': PLOT_ISM_rimPy_neg_rand10,
        'info': 'Negative Reflection + 10cm Randomness',
        'function': calculate_rimpy_rir,
        'params': {
            'reflection_sign': -1,
            'tw_fractional_delay_length': 0,
            'randDist': 0.1
        }
    }
}

# SDN Test Configurations
sdn_tests = {

    '11': {'enabled': False,
              'info': "c [3.87694432 3.91420138 4.00285606 3.99207621 4.02144006 4.01794584]",
              'flags': {
                  'specular_source_injection': True,
                    'injection_c_vector':[3.87694432, 3.91420138, 4.00285606, 3.99207621, 4.02144006, 4.01794584],
              }, 'label': "SDN"},

    '1111': {'enabled': t1,
                  'info': "c1 [1,1,1,1,1,1] src-mic gain:1",
                  'flags': {
                      'specular_source_injection': True,
                        'injection_c_vector': [1, 1, 1, 1, 1,1],
                      'ignore_src_node_atten': True,
                  }, 'label': "SDN"},

    '111': {'enabled': t1,
            'info': "c1 [1,1,1,1,1,1] node-mic gain:1",
            'flags': {
                'specular_source_injection': True,
                'injection_c_vector': [1, 1, 1, 1, 1, 1],
                'ignore_node_mic_atten': True,
            }, 'label': "SDN"},

    '1': {'enabled': t1,
          'info': "c5 [5,1,1,1,1,1]",
          'flags': {
              'specular_source_injection': True,
                'injection_c_vector': [5, 1, 1, 1, 1,1],
              'print_weighted_psk': True,
          }, 'label': "SDN"},

    '2': {'enabled': t2,
        'info': "c5 [5, 5, 1, 1, 1, 1]",
                  'flags': {
                      'specular_source_injection': True,
                      'injection_c_vector': [5, 5, 1, 1, 1,1],  # Example vector
                    'print_weighted_psk': True,
                  }, 'label': "SDN"},

    '3': {'enabled': t3,
        'info': "c5 only src-node gain:1 [5, 1,1,1,1,1]",
                  'flags': {
                      'specular_source_injection': True,
                      'injection_c_vector': [5, 1,1, 1, 1,1],  # Example vector
                'ignore_src_node_atten': True,
                    'print_weighted_psk': True,
                  }, 'label': "SDN"},

    '4': {'enabled': t4,
        'info': "c5 [5, 5, 5, 5, 1]",
                  'flags': {
                      'specular_source_injection': True,
                      'injection_c_vector': [5,5,5,5,5, 1],  # Example vector
                    'print_weighted_psk': True,
                  }, 'label': "SDN"},


    'TestOptimizer': {'enabled': False,
                      'info': "[9.    5.879 9.    9.    9.    9.   ]",
                      'flags': {'specular_source_injection': True,
                                'injection_c_vector': [9., 5.879, 9., 9., 9., 9., ], },
                      'label': "SDN"},

    'TestX': {'enabled': False,
              'info': "c50 , no abs",
              'flags': {'specular_source_injection': True,
                        'source_weighting': 50,
                        'ignore_wall_absorption': True,
                        'ignore_src_node_atten': True,
                        'ignore_node_mic_atten': True,
                         },
              'label': "SDN"},

    'TestY': {'enabled': False,
              # 'absorption': 0.2,
              'info': "[ 7.  7.  7.  7.  7. -5.]",
              'flags': {'specular_source_injection': True,
                        'injection_c_vector': [7., 7., 7., 7., 7., -5.], },
              'label': "SDN"},

    'TestZ': {'enabled': False,
              'info': "[7, 7, -3, -3, -3, 5]",
              'flags': {'specular_source_injection': True,
                        'injection_c_vector': [7, 7, -3, -3, -3, 5], },
              'label': "SDN"},

    'TestT': {'enabled': False,
              'info': "[8, 1, 1, 1, 1, 1]",
              'flags': {'specular_source_injection': True,
                        'injection_c_vector': [8, 1, 1, 1, 1, 1], },
              'label': "SDN"},

    # """ TEST CONFIGURATIONS """
    #
    # """ ORIGINALS """

    'Test1': {'enabled': RUN_SDN_Test1,'info': "c1 orj",'flags': {
                          'specular_source_injection': True,
                          'source_weighting': 1,
                      },'label': "SDN"},

    'Test2': {'enabled': RUN_SDN_Test2, 'info': "c2", 'flags': {
                        'specular_source_injection': True,
                        'source_weighting': 2,
                    }, 'label': "SDN"},

    'Test3': {'enabled': RUN_SDN_Test3,'info': "c3",'flags': {
                      'specular_source_injection': True,
                      'source_weighting': 3,
                  }, 'label': "SDN"},

    'Test4': {'enabled': RUN_SDN_Test4,'info': "c4", 'flags': {
                      'specular_source_injection': True,
                      'source_weighting': 4,
                  }, 'label': "SDN"},

    'Test5': {'enabled': RUN_SDN_Test5, 'info': "c5", 'flags': {
                    'specular_source_injection': True,
                    'source_weighting': 5,
                }, 'label': "SDN Test 5"},

    'Test6': {'enabled': RUN_SDN_Test6, 'info': "c6", 'flags': {
                    'specular_source_injection': True,
                    'source_weighting': 6,
                }, 'label': "SDN Test 6"},

    'Test7': {'enabled': RUN_SDN_Test7, 'info': "c7", 'flags': {
        'specular_source_injection': True,
        'source_weighting': 7,
    }, 'label': "SDN Test 7"},

    # NO ATT CASE (no distance or wall loss)
    'Test1n': {'enabled': RUN_SDN_Test1_noatt, 'info': "c1 no att",'flags': {
                      'specular_source_injection': True,
                      'source_weighting': 1,
                            'ignore_wall_absorption': True,
                            'ignore_src_node_atten': True,
                            'ignore_node_mic_atten': True,
                  },'label': "SDN"},

    'Test2n': {'enabled': RUN_SDN_Test2_noatt, 'info': "c2 no att", 'flags': {
        'specular_source_injection': True,
        'source_weighting': 2,
        'ignore_wall_absorption': True,
        'ignore_src_node_atten': True,
        'ignore_node_mic_atten': True,
    }, 'label': "SDN"},

    'Test3n': {'enabled': RUN_SDN_Test3_noatt, 'info': "c3 no att", 'flags': {
        'specular_source_injection': True,
        'source_weighting': 3,
        'ignore_wall_absorption': True,
        'ignore_src_node_atten': True,
        'ignore_node_mic_atten': True,
    }, 'label': "SDN"},

    'Test4n': {'enabled': RUN_SDN_Test4_noatt, 'info': "c4 no att", 'flags': {
        'specular_source_injection': True,
        'source_weighting': 4,
        'ignore_wall_absorption': True,
        'ignore_src_node_atten': True,
        'ignore_node_mic_atten': True,
    }, 'label': "SDN"},

    'Test5n': {'enabled': RUN_SDN_Test5_noatt, 'info': "c5 no att", 'flags': {
        'specular_source_injection': True,
        'source_weighting': 5,
        'ignore_wall_absorption': True,
        'ignore_src_node_atten': True,
        'ignore_node_mic_atten': True,
    }, 'label': "SDN"},

    'Test6n': {'enabled': RUN_SDN_Test6_noatt, 'info': "c6 no att", 'flags': {
        'specular_source_injection': True,
        'source_weighting': 6,
        'ignore_wall_absorption': True,
        'ignore_src_node_atten': True,
        'ignore_node_mic_atten': True,
    }, 'label': "SDN"},

    'Test4r': {'enabled': RUN_SDN_Test4r,
                          'info': "c4 random c_direction",
                          'flags': {
                              'specular_source_injection_random': True,
                              'source_weighting': 4,
                          }, 'label': "SDN"},

    'Test5r': {'enabled': RUN_SDN_Test5r,
                      'info': "c5 random c_direction",
                      'flags': {
                          'specular_source_injection_random': True,
                          'source_weighting': 5,
                      }, 'label': "SDN"},

    'Test6r': {'enabled': RUN_SDN_Test6r,
                       'info': "c6 random best c_direction",
                       'flags': {
                           'specular_source_injection_random': True,
                           'source_weighting': 6,
                       }, 'label': "SDN"},

    'Test7r': {'enabled': RUN_SDN_Test7r,
                   'info': "c7 random best c_direction",
                   'flags': {
                       'specular_source_injection_random': True,
                       'source_weighting': 7,
                   }, 'label': "SDN"},

    'Test-3np': {
        'enabled': RUN_SDN_Test0,
        'info': "c-3 specular NODE!",
        'flags': {
            'specular_source_injection': True,
            'source_weighting': -3,
            'specular_node_pressure': True,
        },
        'label': "SDN"
    },

    'Test_3': {'enabled': RUN_SDN_Test_3,
               'info': "c-3",
               'flags': {
                   'specular_source_injection': True,
                   'source_weighting': -3,
               }, 'label': "SDN"},

    'Test_3b': {'enabled': RUN_SDN_Test_3b,
                'info': "c-3 SN",
                'flags': {
                    'specular_source_injection': True,
                    'source_weighting': -3,
                    'specular_node_pressure': True,
                }, 'label': "SDN"},

    'Test_3c': {'enabled': RUN_SDN_Test_3c,
                'info': "c-3 SP MAT",
                'flags': {
                    'specular_source_injection': True,
                    'source_weighting': -3,
                    'specular_scattering': True,
                    'specular_increase_coef': 0.01,
                }, 'label': "SDN"},

    'Test_2': {'enabled': RUN_SDN_Test_2,
               'info': "c-2",
               'flags': {
                   'specular_source_injection': True,
                   'source_weighting': -2,
               }, 'label': "SDN"},

    'Test_2b': {'enabled': RUN_SDN_Test_2b,
                'info': "c-2 SN",
                'flags': {
                    'specular_source_injection': True,
                    'source_weighting': -2,
                    'specular_node_pressure': True,
                }, 'label': "SDN"},

    'Test_2c': {'enabled': RUN_SDN_Test_3c,
                'info': "c-2 SP MAT",
                'flags': {
                    'specular_source_injection': True,
                    'source_weighting': -2,
                    'specular_scattering': True,
                    'specular_increase_coef': 0.02,
                }, 'label': "SDN"},


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


    'Test2b': {'enabled': RUN_SDN_Test2b,
               'info': "c2 SN",
               'flags': {
                   'specular_source_injection': True,
                   'source_weighting': 2,
                   'specular_node_pressure': True,
                   'source_pressure_injection_coeff': 1.,
                   'coef': 1 / 5,
               }, 'label': "SDN"},

    'Test2c': {'enabled': RUN_SDN_Test_3c,
               'info': "c2 SP MAT",
               'flags': {
                   'specular_source_injection': True,
                   'source_weighting': 2,
                   'specular_scattering': True,
                   'specular_increase_coef': 0.02,
               }, 'label': "SDN"},

    'Test3r': {'enabled': RUN_SDN_Test3r,
                  'info': "c3 random",
                  'flags': {
                      'specular_source_injection_random': True,
                      'source_weighting': 3,
                  }, 'label': "SDN"},

    'Test3b': {'enabled': RUN_SDN_Test3b,
               'info': "c3 SN",
               'flags': {
                   'specular_source_injection': True,
                   'source_weighting': 3,
                   'specular_node_pressure': True,
                   'source_pressure_injection_coeff': 1.,
                   'coef': 1 / 5,
               }, 'label': "SDN"},

    'Test3c': {'enabled': RUN_SDN_Test_3c,
               'info': "c3 SP MAT",
               'flags': {
                   'specular_source_injection': True,
                   'source_weighting': 3,
                   'specular_scattering': True,
                   'specular_increase_coef': 0.01,
               }, 'label': "SDN"},



    'Test4b': {'enabled': RUN_SDN_Test4b, 'info': "c4 SN", 'flags': {
                   'specular_source_injection': True,
                   'source_weighting': 4,
                   'specular_node_pressure': True,
               }, 'label': "SDN"},

    'Test4c': {'enabled': RUN_SDN_Test4c,
               'info': "c4 SP MAT",
               'flags': {
                   'specular_source_injection': True,
                   'source_weighting': 4,
                   'specular_scattering': True,
                   'specular_increase_coef': 0.01,
               }, 'label': "SDN"},




    'Test5b': {'enabled': RUN_SDN_Test5b,
               'info': "c5 SN",
               'flags': {
                   'specular_source_injection': True,
                   'source_weighting': 5,
                   'specular_node_pressure': True,
               }, 'label': "SDN Test 5"},

    'Test5c': {'enabled': RUN_SDN_Test5c,
               'info': "c5 SP MAT",
               'flags': {
                   'specular_source_injection': True,
                   'source_weighting': 5,
                   'specular_scattering': True,
                   'specular_increase_coef': 0.01,
               }, 'label': "SDN"},



    'Test6b': {'enabled': RUN_SDN_Test6b,
               'info': "c6 SN",
               'flags': {
                   'specular_source_injection': True,
                   'source_weighting': 6,
                   'specular_node_pressure': True,
               }, 'label': "SDN Test 6"},

    'Test6c': {'enabled': RUN_SDN_Test6c,
               'info': "c6 SP MAT",
               'flags': {
                   'specular_source_injection': True,
                   'source_weighting': 6,
                   'specular_scattering': True,
                   'specular_increase_coef': 0.01,
               }, 'label': "SDN"},

    'TestHO_N1': {
        'enabled': RUN_MY_HO_SDN_n1,
        'info': "my ho",
        'flags': {
            'ho_sdn_order': 1,
        # 'ignore_wall_absorption': True,
        # 'ignore_src_node_atten': True,
        # 'ignore_node_mic_atten': True,
        },
        'label': "My ho 2"
    },

    'TestHO_N2_swc5': {
            'enabled': RUN_MY_HO_SDN_n2_swc5,
            'info': "sw-c5-ho-N2",
            'flags': {
                'ho_sdn_order': 2,
        'specular_source_injection': True,
                   'source_weighting': 5,
            },
            'label': ""
        },

    'TestHO_N2_swc3': {
                'enabled': RUN_MY_HO_SDN_n2_swc3,
                'info': "sw-c5-ho-N2",
                'flags': {
                    'ho_sdn_order': 2,
            'specular_source_injection': True,
                       'source_weighting': 3,
                },
                'label': ""
            },

    'TestHO_N3_swc3': {
                    'enabled': RUN_MY_HO_SDN_n3_swc3,
                    'info': "sw-c5-ho-N3",
                    'flags': {
                        'ho_sdn_order': 3,
                'specular_source_injection': True,
                           'source_weighting': 3,
                    },
                    'label': ""
                },

    'TestHO_N1_noatt': {
            'enabled': RUN_MY_HO_SDN_n1noatt,
            'info': "my ho 1 no att",
            'flags': {
                'ho_sdn_order': 1,
                'ignore_wall_absorption': True,
                'ignore_src_node_atten': True,
                'ignore_node_mic_atten': True,
            },
            'label': ""
        },

    'TestHO_N2_noatt': {
        'enabled': RUN_MY_HO_SDN_n2noatt,
        'info': "my ho 2 no att",
        'flags': {
            'ho_sdn_order': 2,
            'ignore_wall_absorption': True,
            'ignore_src_node_atten': True,
            'ignore_node_mic_atten': True,

        },
        'label': ""
    },

}

# HO-SDN Test Configurations
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


# Function to run SDN tests with given configuration, new implementation (SDN-Ege)
def run_sdn_test(test_name, config):
    # Store original signal
    original_signal = room.source.signal

    # Check if a specific source signal is requested
    if 'source_signal' in config:
        if config['source_signal'] == 'gaussian':
            room.source.signal = impulse_gaussian['signal']
            print(f"Using Gaussian impulse for {test_name}")
        elif config['source_signal'] == 'dirac':
            room.source.signal = impulse_dirac['signal']
            print(f"Using Dirac impulse for {test_name}")

    sdn, rir, label, is_default = calculate_sdn_rir(room_parameters, test_name, room, duration, Fs, config)

    if 'source_signal' in config:
        room.source.signal = original_signal

    rirs[label] = rir

    # Track if this is a default configuration
    if is_default:
        default_rirs.add(label)

    return sdn


# Function to run HO-SDN tests with given configuration
def run_ho_sdn_test(test_name, config):
    # Store original signal
    original_signal = room.source.signal

    # Check if a specific source signal is requested
    if 'source_signal' in config:
        if config['source_signal'] == 'gaussian':
            room.source.signal = impulse_gaussian['signal']
            print(f"Using Gaussian impulse for {test_name}")
        elif config['source_signal'] == 'dirac':
            room.source.signal = impulse_dirac['signal']
            print(f"Using Dirac impulse for {test_name}")

    # Get order from config
    order = config.get('order')  # Default to order 2 if not specified

    # Calculate HO-SDN RIR
    rir, label = calculate_ho_sdn_rir(room_parameters, Fs, duration, config['source_signal'], order=order)

    # Restore original signal
    if 'source_signal' in config:
        room.source.signal = original_signal

    # Add info to label if provided
    if 'info' in config and config['info']:
        label = f"{label}: {config['info']}"

    rirs[label] = rir

    return rir, label


def get_method_pairs():
    """Get pairs of methods with their original labels and configurations.
    Returns a list of dictionaries with clean method names and their configurations."""
    pairs = []

    # Get actual labels from rirs dictionary
    method_labels = list(rirs.keys())

    # Generate pairs with configurations
    for i in range(len(method_labels)):
        for j in range(i + 1, len(method_labels)):
            label1 = method_labels[i]
            label2 = method_labels[j]

            # Get base labels (before the colon)
            base1 = label1.split(': ')[0] if ': ' in label1 else label1
            base2 = label2.split(': ')[0] if ': ' in label2 else label2

            # Extract configuration info if available (after the colon)
            info1 = label1.split(': ')[1] if ': ' in label1 else ''
            info2 = label2.split(': ')[1] if ': ' in label2 else ''

            # Get binary flags from the label // new part
            flags1 = {}
            flags2 = {}

            # Extract flags from the label
            if 'Test' in base1:
                test_name = base1.split('Test')[1].split()[0]  # Get test number
                if test_name in sdn_tests:
                    flags1 = sdn_tests[test_name]['flags']

            if 'Test' in base2:
                test_name = base2.split('Test')[1].split()[0]  # Get test number
                if test_name in sdn_tests:
                    flags2 = sdn_tests[test_name]['flags']

            # Combine configurations and flags
            combined_info = ""
            if info1 and info2:
                combined_info = f"{info1} vs {info2}"
            elif info1:
                combined_info = info1
            elif info2:
                combined_info = info2

            # Add flag differences to combined_info // new part
            flag_diffs = []
            all_flags = set(flags1.keys()) | set(flags2.keys())
            for flag in all_flags:
                if flag not in flags1 or flag not in flags2 or flags1[flag] != flags2[flag]:
                    flag_diffs.append(f"{flag}={flags1.get(flag, 'False')} vs {flags2.get(flag, 'False')}")

            if flag_diffs:
                if combined_info:
                    combined_info += " | "
                combined_info += " | ".join(flag_diffs)

            pairs.append({
                'pair': f"{base1} vs {base2}",
                'label1': label1,
                'label2': label2,
                'info': combined_info
            })

    return pairs


if __name__ == '__main__':

    if not PICKLE_LOAD_RIRS:
        # Loop through ISM/rimPy methods
        for method_name, config in ism_methods.items():
            if config['enabled']:
                print(f"--- Running {method_name}: {config.get('info', '')} ---")
                # Call the respective function with its parameters
                rir, label = config['function'](
                    room_parameters,
                    duration,
                    Fs,
                    **config['params']
                )
                rirs[label] = rir

        if duration > 0.7:
            # Theoretical RT60 if room dimensions and absorption are provided
            if room_dim is not None and room_parameters['absorption'] is not None:
                rt60_sabine, rt60_eyring = pp.calculate_rt60_theoretical(room_dim, room_parameters['absorption'])
                print(f"\nTheoretical RT60 values of the room with a= {room_parameters['absorption']}:")
                print(f"Sabine: {rt60_sabine:.3f} s")
                print(f"Eyring: {rt60_eyring:.3f} s")

        # Calculate SDN-Base RIR
        if PLOT_SDN_BASE:
            from archive.sdn_base import calculate_sdn_base_rir

            sdn_base_rir = calculate_sdn_base_rir(room_parameters, duration, Fs)
            label = 'SDN-Base (original)'
            rirs[label] = sdn_base_rir

        # Run SDN tests
        for test_name, config in sdn_tests.items():
            if config['enabled']:
                sdn = run_sdn_test(test_name, config)

        # Run HO-SDN tests
        for test_name, config in ho_sdn_tests.items():
            if config['enabled']:
                rir, label = run_ho_sdn_test(test_name, config)

        # Normalize all RIRs

        rirs = rir_normalisation(rirs, room, Fs, normalize_to_first_impulse)

    else:  # rirs are loaded from pickle:
        with open(file_name, 'rb') as f:
            rirs = pickle.load(f)

    # Calculate reflection arrival times if needed
    reflection_times = None
    if PLOT_REFLECTION_LINES:
        path_tracker = path_tracker.PathTracker()
        sdn_calc = SDNCalculator(room.walls, room.source.srcPos, room.micPos)
        ism_calc = ISMCalculator(room.walls, room.source.srcPos, room.micPos)
        sdn_calc.set_path_tracker(path_tracker)
        ism_calc.set_path_tracker(path_tracker)

        # Compare paths and analyze invalid ISM paths (without printing)
        PathCalculator.compare_paths(sdn_calc, ism_calc, max_order=3, print_comparison=False)
        ism_calc.analyze_paths(max_order=3, print_invalid=False)

        # Get arrival times for each order
        arrival_times = path_tracker.get_latest_arrival_time_by_order('ISM')
        reflection_times = {
            'first_order': arrival_times.get(1),
            'second_order': arrival_times.get(2),
            'third_order': arrival_times.get(3)
        }

        if pulse_analysis == "upto_4":
            # Get the arrival time for 3rd order reflections
            arrival_time_3rd = arrival_times.get(4)  # in seconds
            if arrival_time_3rd is not None:
                sample_idx_3rd = int(arrival_time_3rd * Fs)  # convert to samples

                # Count nonzero samples for SDN and ISM up to 3rd order arrival
                threshold = 0  # threshold for considering a sample nonzero

                # For SDN original
                sdn_rir = rirs['SDN-Original:  ']
                sdn_nonzero = np.sum(np.abs(sdn_rir[:sample_idx_3rd]) > threshold)
                print(f"\nSDN nonzero samples up to 4rd order ({arrival_time_3rd:.3f}s): {sdn_nonzero}")

                # For ISM rimPy negative

                ism_rir = rirs['ISM-rimpy-negREF']
                ism_nonzero = np.sum(np.abs(ism_rir[:sample_idx_3rd]) > threshold)
                print(f"ISM-rimpy-neg nonzero samples up to 4rd order ({arrival_time_3rd:.3f}s): {ism_nonzero}")

                # Print valid ISM paths count
                path_tracker.print_valid_paths_count(4)

    if interactive_rirs:
        reversed_rirs = dict(reversed(list(rirs.items())))
        if UNIFIED_PLOTS:
            pp.create_unified_interactive_plot(reversed_rirs, Fs, room_parameters,
                                               reflection_times=reflection_times)
            plt.show(block=False)
        else:
            import importlib
            from analysis import plot_room as pp
            importlib.reload(pp)
            pp.create_interactive_rir_plot(rirs, Fs)
            plt.show(block=False)

    # Calculate RT60 values for all RIRs
    if duration > 0.7:

        rt60_values = {}
        for label, rir in rirs.items():
            # rt60_values[label] = pp.calculate_rt60_from_rir(rir, Fs, plot=False)
            rt60_values[label] = an.calculate_rt60_from_rir(rir, Fs, plot=False)

        print("\nReverberation Time Analysis:")
        print("-" * 50)
        print("\nMeasured RT60 values:")
        for label, rt60 in rt60_values.items():
            print(f"{label}: {rt60:.3f} s")

    if PLOT_EDC:
        if not UNIFIED_PLOTS:
            pp.create_interactive_edc_plot(rirs, Fs)

    if PLOT_NED and not UNIFIED_PLOTS:
        plt.figure(figsize=(12, 6))
        print("old NED plot")
        # Plot both normalized and raw echo densities
        for label, rir in rirs.items():
            # Normalized Echo Density
            echo_density = ned.echoDensityProfile(rir, fs=Fs)
            # Create time array in seconds
            time_axis = np.arange(len(echo_density)) / Fs
            plt.plot(time_axis, echo_density, label=label, alpha=0.7)

        # Configure plots
        plt.title('Normalized Echo Density')
        plt.xlabel('Time (s)')
        plt.ylabel('Normalized Echo Density')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

    if PLOT_lsd:
        # Get all unique pairs of RIR labels
        rir_labels = list(rirs.keys())
        i = 0
        for j in range(i + 1, len(rir_labels)):
            label1 = rir_labels[i]
            label2 = rir_labels[j]
            rir1 = rirs[label1]
            rir2 = rirs[label2]
            lsd_values = an.compute_LSD(rir1, rir2, Fs)
            print(f"{label1} vs {label2}: {lsd_values:.2f} dB LSD difference")
            an.plot_spectral_comparison(rir1, rir2, Fs=Fs, label1=label1, label2=label2)

    if PLOT_FREQ:
        # Calculate and plot frequency responses for all RIRs
        for rir_label, rir in rirs.items():
            # Add RT60 to the label
            plot_label = f"{rir_label} (RT60: {rt60_values[rir_label]:.3f}s)"
            freq, magnitude = ff.calculate_frequency_response(rir, Fs)
            ff.plot_frequency_response(freq, magnitude, label=f'FRF {plot_label}')

    # Process method pairs once
    method_pairs = get_method_pairs()

    if Print_RIR_comparison_metrics:
        # Compute all RIR metrics in batch (moved to analysis.py)
        rirs_analysis = an.compute_rir_metrics_batch(rirs, Fs)
        
        # Print individual RIR metrics
        an.print_rir_metrics(rirs_analysis)
        
        # Store comparison results for both energy and smoothed signals
        comparison_results = {
            'early_energy': [],
            'smoothed_energy': []
        }
        
        # Compare energy signals between all pairs
        energy_comparisons = an.compare_rir_pairs(rirs_analysis, method_pairs, comparison_type='early_energy')
        comparison_results['early_energy'] = energy_comparisons
        an.print_comparison_results(energy_comparisons, "Energy (squared RIRs) Signal Comparison Results")
        
        # Compare smoothed signals between all pairs
        smoothed_comparisons = an.compare_rir_pairs(rirs_analysis, method_pairs, comparison_type='smoothed_energy')
        comparison_results['smoothed_energy'] = smoothed_comparisons
        an.print_comparison_results(smoothed_comparisons, "Smoothed (hann windowed squared RIRs) Signal Comparison Results")
        
        print("\n")
        
        # Compare EDCs between all pairs
        edc_comparisons = an.compare_edc_pairs(rirs, get_method_pairs(), Fs)
        an.print_edc_comparisons(edc_comparisons)

    if plot_smoothed_rirs:
        # Iterate through each method in rirs_analysis
        for rir_label, signals in rirs_analysis.items():
            plt.figure(figsize=(12, 6))

            # Plot Energy Signal
            plt.plot(signals["energy"], label=f'Energy Signal - {rir_label}', color='blue')

            # Plot Smoothed Signal
            plt.plot(signals["smoothed_energy"], label=f'Smoothed Energy Signal - {rir_label}', color='orange')

            # Add titles and labels
            plt.title(f'Energy and Smoothed Energy Signals for {rir_label}')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.grid()
            plt.legend()

            # Show the plot
            plt.show()

# Only Path Length Analysis, No RIR Calculation
if ISM_SDN_PATH_DIFF_TABLE:

    if not PLOT_REFLECTION_LINES:
        path_tracker = path_tracker.PathTracker()
        sdn_calc = SDNCalculator(room.walls, room.source.srcPos, room.micPos)
        ism_calc = ISMCalculator(room.walls, room.source.srcPos, room.micPos)
        sdn_calc.set_path_tracker(path_tracker)
        ism_calc.set_path_tracker(path_tracker)

    PathCalculator.compare_paths(sdn_calc, ism_calc, max_order=3, print_comparison=True)
    # analyze_paths() returns a list of invalid paths (only for ISM calculator)
    invalid_paths = ism_calc.analyze_paths(max_order=3, print_invalid=True)

    # Visualize example ISM paths
    example_paths = [
        # ['s', 'east', 'west', 'm'],
        # ['s', 'west', 'm'],
        # ['s', 'ceiling', 'm'],
        # ['s', 'm'],
        # ['s', 'west', 'east', 'north', 'm']
    ]

    for path in example_paths:
        pp.plot_ism_path(room, ism_calc, path)
        plt.show()

if PLOT_ROOM:
    if not ISM_SDN_PATH_DIFF_TABLE:
        pp.plot_room(room)

if pulse_analysis == "all":
    # Pulse analysis moved to analysis.py for better organization
    pulse_results = an.analyze_rir_pulses(rirs, Fs, print_results=True)

# import pickle
# # # Save the RIRs to a file
#
# try:
#     with open(file_name, 'rb') as f:
#         existing_rirs = pickle.load(f)
#     # If the file exists, write a message
#     print("File already exists. Not overwriting. Please delete the file to save new data.")
#
# # dump the file otherwise
# except FileNotFoundError:
#     # If the file doesn't exist, write the new data
#     print("File not found. Saving new data.")
#     # Save the RIRs to the file
#     with open(file_name, 'wb') as f:
#         # Save the RIRs to the file
#         pickle.dump(rirs, f)


if SAVE_audio:
    # Save the RIRs as wav
    import soundfile as sf

    for rir_label, rir_audio in rirs.items():
        sf.write(f"{rir_label}.wav", rir_audio * 0.8, Fs)

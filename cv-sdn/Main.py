# Python Modules:
import time
import math

# Downloaded Modules:
import numpy as np
import matplotlib.pyplot as plt

# Created Modules:
import Coupled_SDN
import Simulation_plot

SPEED_OF_SOUND = 343


def initialization(first_room, second_room):
    first_room.second_room = second_room

    second_room.find_images()
    first_room.find_images()

    second_room.find_sdn_nodes()
    first_room.find_sdn_nodes()

    first_room.create_delay_lines()
    second_room.create_delay_lines()

    first_room.find_distances()
    second_room.find_distances()


def plot(first_room, second_room, mic, src, door, is_plot_on, is_images_shown, is_sdn_shown):
    # to-do: remove this link, directly call plot function.

    Simulation_plot.plot(first_room, second_room, mic, src, door, is_plot_on, is_images_shown, is_sdn_shown)


def main(area, a, b,max_iteration, destination_path):
    # Beginning of the program:
    t0 = time.time()

    # Set up Door:
    # Door positions: (5.6, 3.8, 1.0)
    #                 (5.6, 3.5, 1.3)
    #                 (5.6, 3.0, 1.8)
    #                 (5.6, 2.8, 2)
    #                 (5.6, 2.6, 2.2)
    #                 (5.6, 2.4, 2.4)
    door = Coupled_SDN.Door(np.array([5.6, 3.0, 1.8]), area)

    # Set up Microphone:
    # mic = Coupled_SDN.Microphone(np.array([11, 1.40, 1]))
    # For C3 Position use (2.6,2.9,1)
    # For original Position use (3.6,3.9,1)
    mic = Coupled_SDN.Microphone(np.array([3.6, 3.9, 1]))
    # Set up Source
    # src = Coupled_SDN.Source(np.array([1, 3.9, 1.2]))
    # For original position use src = Coupled_SDN.Source(np.array([0.8, 2.2, 2.7]))
    src = Coupled_SDN.Source(np.array([0.8, 2.2, 2.7]))

    # Set up Room
    # For original room use     first_room = Coupled_SDN.Room(5.6, 4.8, 6.4,
    #                                   a,
    #                                   mic,
    #                                   src,
    #                                   door,
    #                                   1)
    first_room = Coupled_SDN.Room(6, 5, 3,
                                  a,
                                  mic,
                                  src,
                                  door,
                                  1)
    # For original room use     second_room = Coupled_SDN.SecondRoom(6.8, 7.2, 7,
    #                                          b,
    #                                          None,
    #                                          None,
    #                                          door,
    #                                          first_room)
    second_room = Coupled_SDN.SecondRoom(20,3,3,
                                         b,
                                         None,
                                         None,
                                         door,
                                         first_room)

    # door = Coupled_SDN.Door(np.array([11.2, 7, 2.6]), area)
    #
    # # Set up Microphone:
    # # mic = Coupled_SDN.Microphone(np.array([11, 1.40, 1]))
    # # For C3 Position use (2.6,2.9,1)
    # # For original Position use (3.6,3.9,1)
    # mic = Coupled_SDN.Microphone(np.array([1.2, 3.8, 2]))
    # # Set up Source
    # # src = Coupled_SDN.Source(np.array([1, 3.9, 1.2]))
    # src = Coupled_SDN.Source(np.array([1.6, 4.4, 5.4]))
    #
    # # Set up Room
    # first_room = Coupled_SDN.Room(11.2, 9.6, 12.8,
    #                               a,
    #                               mic,
    #                               src,
    #                               door)
    #
    # second_room = Coupled_SDN.SecondRoom(13.6, 14.4, 14,
    #                                      b,
    #                                      None,
    #                                      None,
    #                                      door,
    #                                      first_room)
    # Initialize the system
    initialization(first_room, second_room)

    # Plot the system
    # plot(first_room, second_room, mic, src, door, True, True, True)

    # Read File and input to source
    read_write = Coupled_SDN.SoundFileRW()
    read_write.read_sound_file(".\\samples\\tick.wav", src, 1)

    t1 = time.time()
    print("Initialization is finished in ", float("{0:.2f}".format(t1 - t0)), " seconds!")
    print("Running Coupled SDN algorithm.Please wait...")

    # Coupled-SDN Algorithm
    for i in range(0, max_iteration):
        first_room.tick_function()

    rate, data = read_write.write_sound_file(".\\samples\\" + destination_path, mic)
    t2 = time.time()
    print("Coupled SDN algorithm is finished in", float("{0:.2f}".format(t2 - t1)), " seconds!")

    return rate, data


def edc_curve(rate, data, color):
    sampling_rate = rate

    # time_passed_one = np.arange(0, len(data) / sampling_rate, (1 / sampling_rate))
    #
    # plt.plot(time_passed_one, data)
    # plt.show()

    sampling_rate = float(sampling_rate)
    time_passed = np.arange(0, len(data) / sampling_rate, (1 / sampling_rate))

    signal_energy = (np.cumsum(data[::-1] ** 2) / np.sum(data[::-1]))[::-1]
    signal_decibel = 10.0 * np.log10(signal_energy / np.max(signal_energy))

    # Enable the following part for echo density:
    # signal_ned = np.zeros(len(data))
    #
    # window_size = round(44100 * 0.01)
    # window = np.hanning(window_size)
    # window = window / sum(window)
    # window_length = len(window)
    #
    # for i in range(0, len(data) - window_length):
    #     values = data[i:(i + window_length)]
    #     sigma = math.sqrt(sum(np.multiply(window, np.square(values))))
    #
    #     sumValues = 0
    #     for value in values:
    #         if abs(value) > sigma:
    #             sumValues += window[values.tolist().index(value)]
    #
    #     signal_ned[i] = (1 / math.erfc(1 / math.sqrt(2))) * sumValues

    # plt.axis([0, 0.2, -0.1, 1.2])
    plt.plot(time_passed, signal_decibel, ls='solid', color=color, label='Energy decay curve', linewidth=1)
    # plt.plot(time_passed, data, ls='solid', color=color, label='Energy decay curve', linewidth=1)
    # plt.plot(time_passed, signal_ned, ls='solid', color='r', label='Energy decay curve', linewidth=1)


if __name__ == '__main__':
    # rate, data = main(93.6, 0.25, 0.02)
    # edc_curve(rate, data, 'g')
    # rate, data = main(46.8, 0.25, 0.02)
    # edc_curve(rate, data, 'b')
    # rate, data = main(23.4, 0.25, 0.02)
    # edc_curve(rate, data, 'r')
    # rate, data = main(11.7, 0.25, 0.02)
    # edc_curve(rate, data, 'y')
    # rate, data = main(4.68, 0.25, 0.02)
    # edc_curve(rate, data, 'c')
    # rate, data = main(2.34, 0.25, 0.02)
    # edc_curve(rate, data, 'c')
    # rate, data = main(18.4, 0.4, 0.17)
    # edc_curve(rate, data, 'b')
    # rate, data = main(2.25, 0.32, 0.13)
    # edc_curve(rate, data, 'c')

    # Aperture Areas = (2.25, 4.62, 9.18, 12.28, 15.36, 18.4)

    # rate, data = main(4.62, 0.01, 0.01,661500, "results\\15%_1kHz_0.01-0.01.wav")
    # # rate, data = main(4.62, 0.02, 0.01, "results\\15%_1kHz_0.02-0.01.wav")
    # rate, data = main(4.62, 0.03, 0.01, 573300, "results\\15%_1kHz_0.03-0.01.wav")
    # rate, data = main(4.62, 0.05, 0.01, 100000, "results\\15%_1kHz_0.05-0.01.wav")
    # rate, data = main(4.62, 0.1, 0.01, 80000, "results\\15%_1kHz_0.1-0.01.wav")
    # rate, data = main(4.62, 0.2, 0.01, 44100, "results\\15%_1kHz_0.2-0.01.wav")
    # rate, data = main(4.62, 0.3, 0.01, 44100, "results\\15%_1kHz_0.3-0.01.wav")
    # rate, data = main(4.62, 0.4, 0.01, 44100, "results\\15%_1kHz_0.4-0.01.wav")
    # rate, data = main(4.62, 0.5, 0.01, 44100, "results\\15%_1kHz_0.5-0.01.wav")
    # rate, data = main(4.62, 0.6, 0.01, 44100, "results\\15%_1kHz_0.6-0.01.wav")
    # rate, data = main(4.62, 0.7, 0.01, 44100, "results\\15%_1kHz_0.7-0.01.wav")
    # rate, data = main(4.62, 0.8, 0.01,44100 , "results\\15%_1kHz_0.8-0.01.wav")

    # rate, data = main(9.18, 0.01, 0.01,661500, "results\\30%_1kHz_0.01-0.01.wav")
    # # rate, data = main(4.62, 0.02, 0.01, "results\\15%_1kHz_0.02-0.01.wav")
    # rate, data = main(9.18, 0.03, 0.01, 573300, "results\\30%_1kHz_0.03-0.01.wav")
    # rate, data = main(9.18, 0.05, 0.01, 100000, "results\\30%_1kHz_0.05-0.01.wav")
    # rate, data = main(9.18, 0.1, 0.01, 80000, "results\\30%_1kHz_0.1-0.01.wav")
    # rate, data = main(9.18, 0.2, 0.01, 44100, "results\\30%_1kHz_0.2-0.01.wav")
    # rate, data = main(9.18, 0.3, 0.01, 44100, "results\\30%_1kHz_0.3-0.01.wav")
    # rate, data = main(9.18, 0.4, 0.01, 44100, "results\\30%_1kHz_0.4-0.01.wav")
    # rate, data = main(9.18, 0.5, 0.01, 44100, "results\\30%_1kHz_0.5-0.01.wav")
    # rate, data = main(9.18, 0.6, 0.01, 44100, "results\\30%_1kHz_0.6-0.01.wav")
    # rate, data = main(9.18, 0.7, 0.01, 44100, "results\\30%_1kHz_0.7-0.01.wav")
    # rate, data = main(9.18, 0.8, 0.01,44100 , "results\\30%_1kHz_0.8-0.01.wav")

    rate, data = main(2, 0.01, 0.01,661500, "secondScenario_results\\15%_1kHz_0.01-0.01.wav")
    rate, data = main(2, 0.02, 0.01, "results\\15%_1kHz_0.02-0.01.wav")
    rate, data = main(2, 0.03, 0.01, 573300, "secondScenario_results\\15%_1kHz_0.03-0.01.wav")
    rate, data = main(2, 0.05, 0.01, 100000, "secondScenario_results\\15%_1kHz_0.05-0.01.wav")
    rate, data = main(2, 0.1, 0.01, 80000, "secondScenario_results\\15%_1kHz_0.1-0.01.wav")
    rate, data = main(2, 0.2, 0.01, 44100, "secondScenario_results\\15%_1kHz_0.2-0.01.wav")
    rate, data = main(2, 0.3, 0.01, 44100, "secondScenario_results\\15%_1kHz_0.3-0.01.wav")
    rate, data = main(2, 0.4, 0.01, 44100, "secondScenario_results\\15%_1kHz_0.4-0.01.wav")
    rate, data = main(2, 0.5, 0.01, 44100, "secondScenario_results\\15%_1kHz_0.5-0.01.wav")
    rate, data = main(2, 0.6, 0.01, 44100, "secondScenario_results\\15%_1kHz_0.6-0.01.wav")
    rate, data = main(2, 0.7, 0.01, 44100, "secondScenario_results\\15%_1kHz_0.7-0.01.wav")
    rate, data = main(2, 0.8, 0.01,44100 , "secondScenario_results\\15%_1kHz_0.8-0.01.wav")

    # rate, data = main(9.18, 0.01, 0.01,661500, "results\\30%_1kHz_0.01-0.01.wav")
    # # rate, data = main(4.62, 0.02, 0.01, "results\\15%_1kHz_0.02-0.01.wav")
    # rate, data = main(9.18, 0.03, 0.01, 573300, "results\\30%_1kHz_0.03-0.01.wav")
    # rate, data = main(9.18, 0.05, 0.01, 100000, "results\\30%_1kHz_0.05-0.01.wav")
    # rate, data = main(9.18, 0.1, 0.01, 80000, "results\\30%_1kHz_0.1-0.01.wav")
    # rate, data = main(9.18, 0.2, 0.01, 44100, "results\\30%_1kHz_0.2-0.01.wav")
    # rate, data = main(9.18, 0.3, 0.01, 44100, "results\\30%_1kHz_0.3-0.01.wav")
    # rate, data = main(9.18, 0.4, 0.01, 44100, "results\\30%_1kHz_0.4-0.01.wav")
    # rate, data = main(9.18, 0.5, 0.01, 44100, "results\\30%_1kHz_0.5-0.01.wav")
    # rate, data = main(9.18, 0.6, 0.01, 44100, "results\\30%_1kHz_0.6-0.01.wav")
    # rate, data = main(9.18, 0.7, 0.01, 44100, "results\\30%_1kHz_0.7-0.01.wav")
    # rate, data = main(9.18, 0.8, 0.01,44100 , "results\\30%_1kHz_0.8-0.01.wav")

    # edc_curve(rate, data, 'c')
    # plt.show()

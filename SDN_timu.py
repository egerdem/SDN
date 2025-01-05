import numpy as np
import matplotlib.pyplot as plt
import collections as col
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import time

from scipy.integrate import trapz
from scipy import signal as sig
from scipy.io import wavfile

import processor

# Constants:
SCATTERING_MATRIX = (2 / 5 * np.ones((5, 5)) - np.identity(5))

print(SCATTERING_MATRIX)

SAMPLING_RATE = 44100
SPEED_OF_SOUND = 343


class Room(object):
    def __init__(self, x, y, z
                 , absorption_coefficient
                 , microphone
                 , source):

        if microphone.x > x or microphone.y > y or microphone.z > z \
                or microphone.x < 0 or microphone.y < 0 or microphone.z < 0:
            raise AssertionError("Microphone is out of bounds!")
        if source.x > x or source.y > y or source.z > z \
                or source.x < 0 or source.y < 0 or source.z < 0:
            raise AssertionError("Source is out of bounds!")

        self.x = np.array([0., x])
        self.y = np.array([0., y])
        self.z = np.array([0., z])

        self.microphone = microphone
        self.source = source

        self.images = []
        self.nodes = []

        self.WALL_REFLECTION_COEFFICIENT = math.sqrt(1 - absorption_coefficient)

    def find_images(self):

        # Image Method
        source = self.source

        self.images.append(np.array([-source.x, source.y, source.z]))
        self.images.append(np.array([2 * self.x[1] - source.x, source.y, source.z]))

        self.images.append(np.array([source.x, -source.y, source.z]))
        self.images.append(np.array([source.x, 2 * self.y[1] - source.y, source.z]))

        self.images.append(np.array([source.x, source.y, -source.z]))
        self.images.append(np.array([source.x, source.y, 2 * self.z[1] - source.z]))

    def find_sdn_nodes(self):

        microphone = self.microphone

        x = self.x[1]
        y = self.y[1]
        z = self.z[1]

        for image in self.images:
            # to-do: Find a simpler way of performing SDN_Calculation
            if image[0] < 0:

                # Plane points and normal vector
                point1 = np.array([0, random.uniform(0, y), random.uniform(0, z)])
                point2 = np.array([0, random.uniform(0, y), random.uniform(0, z)])
                point3 = np.array([0, random.uniform(0, y), random.uniform(0, z)])

                vector1 = point3 - point1
                vector2 = point2 - point1

                plane_normal = np.cross(vector1, vector2)

                # Line point and direction
                line_direction = np.array([image[0] - microphone.x, image[1] - microphone.y, image[2] - microphone.z])

                line_point = np.array([microphone.x, microphone.y, microphone.z])

                # Calculation of intersection

                ndotu = plane_normal.dot(line_direction)
                w = line_point - point1
                si = -plane_normal.dot(w) / ndotu
                Psi = w + si * line_direction + point1

                new_Node = SDN_Node(Psi[0], Psi[1], Psi[2], "Node1")
                new_Node.index = 1
                self.nodes.append(new_Node)
            elif image[1] > y:
                # Plane points and normal vector
                point1 = np.array([random.uniform(0, x), y, random.uniform(0, z)])
                point2 = np.array([random.uniform(0, x), y, random.uniform(0, z)])
                point3 = np.array([random.uniform(0, x), y, random.uniform(0, z)])

                vector1 = point3 - point1
                vector2 = point2 - point1

                plane_normal = np.cross(vector1, vector2)

                # Line point and direction
                line_direction = np.array([image[0] - microphone.x, image[1] - microphone.y, image[2] - microphone.z])

                line_point = np.array([microphone.x, microphone.y, microphone.z])

                # Calculation of intersection

                ndotu = plane_normal.dot(line_direction)
                w = line_point - point1
                si = -plane_normal.dot(w) / ndotu
                Psi = w + si * line_direction + point1
                new_Node = SDN_Node(Psi[0], Psi[1], Psi[2], "Node2")
                new_Node.index = 2
                self.nodes.append(new_Node)
            elif image[0] > x:
                # Plane points and normal vector
                point1 = np.array([x, random.uniform(0, y), random.uniform(0, z)])
                point2 = np.array([x, random.uniform(0, y), random.uniform(0, z)])
                point3 = np.array([x, random.uniform(0, y), random.uniform(0, z)])

                vector1 = point3 - point1
                vector2 = point2 - point1

                plane_normal = np.cross(vector1, vector2)

                # Line point and direction
                line_direction = np.array([image[0] - microphone.x, image[1] - microphone.y, image[2] - microphone.z])

                line_point = np.array([microphone.x, microphone.y, microphone.z])

                # Calculation of intersection

                ndotu = plane_normal.dot(line_direction)
                w = line_point - point1
                si = -plane_normal.dot(w) / ndotu
                Psi = w + si * line_direction + point1
                new_Node = SDN_Node(Psi[0], Psi[1], Psi[2], "Node3")
                new_Node.index = 3
                self.nodes.append(new_Node)
            elif image[1] < 0:
                # Plane points and normal vector
                point1 = np.array([random.uniform(0, x), 0, random.uniform(0, z)])
                point2 = np.array([random.uniform(0, x), 0, random.uniform(0, z)])
                point3 = np.array([random.uniform(0, x), 0, random.uniform(0, z)])

                vector1 = point3 - point1
                vector2 = point2 - point1

                plane_normal = np.cross(vector1, vector2)

                # Line point and direction
                line_direction = np.array([image[0] - microphone.x, image[1] - microphone.y, image[2] - microphone.z])

                line_point = np.array([microphone.x, microphone.y, microphone.z])

                # Calculation of intersection

                ndotu = plane_normal.dot(line_direction)
                w = line_point - point1
                si = -plane_normal.dot(w) / ndotu
                Psi = w + si * line_direction + point1
                new_Node = SDN_Node(Psi[0], Psi[1], Psi[2], "Node4")
                new_Node.index = 4
                self.nodes.append(new_Node)
            elif image[2] < 0:

                # Plane points and normal vector
                point1 = np.array([random.uniform(0, x), random.uniform(0, y), 0])
                point2 = np.array([random.uniform(0, x), random.uniform(0, y), 0])
                point3 = np.array([random.uniform(0, x), random.uniform(0, y), 0])

                vector1 = point3 - point1
                vector2 = point2 - point1

                plane_normal = np.cross(vector1, vector2)

                # Line point and direction
                line_direction = np.array([image[0] - microphone.x, image[1] - microphone.y, image[2] - microphone.z])

                line_point = np.array([microphone.x, microphone.y, microphone.z])

                # Calculation of intersection

                ndotu = plane_normal.dot(line_direction)
                w = line_point - point1
                si = -plane_normal.dot(w) / ndotu
                Psi = w + si * line_direction + point1
                new_Node = SDN_Node(Psi[0], Psi[1], Psi[2], "Node5")
                new_Node.index = 5
                self.nodes.append(new_Node)
            elif image[2] > z:
                # Plane points and normal vector
                point1 = np.array([random.uniform(0, x), random.uniform(0, y), z])
                point2 = np.array([random.uniform(0, x), random.uniform(0, y), z])
                point3 = np.array([random.uniform(0, x), random.uniform(0, y), z])

                vector1 = point3 - point1
                vector2 = point2 - point1

                plane_normal = np.cross(vector1, vector2)

                # Line point and direction
                line_direction = np.array([image[0] - microphone.x, image[1] - microphone.y, image[2] - microphone.z])

                line_point = np.array([microphone.x, microphone.y, microphone.z])

                # Calculation of intersection

                ndotu = plane_normal.dot(line_direction)
                w = line_point - point1
                si = -plane_normal.dot(w) / ndotu
                Psi = w + si * line_direction + point1
                new_Node = SDN_Node(Psi[0], Psi[1], Psi[2], "Node6")
                new_Node.index = 6
                self.nodes.append(new_Node)
            else:
                pass
        # Order SDN nodes by their indexes
        myOrder = [0, 3, 1, 2, 4, 5]
        self.nodes = [self.nodes[i] for i in myOrder]

    def Plot(self, is_images_shown, is_sdn_shown):

        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')

        # Get Values from the class:
        x = self.x[1]
        y = self.y[1]
        z = self.z[1]
        ax = self.ax
        microphone = self.microphone
        source = self.source

        # Set limits of the plot
        ax.set_xlim([-max(x, y, z) * 2, max(x, y, z) * 2])
        ax.set_ylim([-max(x, y, z) * 2, max(x, y, z) * 2])
        ax.set_zlim([-max(x, y, z) * 2, max(x, y, z) * 2])

        # Set Labels of the plot
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')

        # Draw microphone and source
        Axes3D.scatter(self.ax, microphone.x, microphone.y, microphone.z, color='b')
        Axes3D.scatter(self.ax, source.x, source.y, source.z, color='r')

        # to-do: use for loops when creating plot
        # Draw Edges
        dimensions = np.array([[0, 0], [x, x], [y, y], [z, z]])
        Axes3D.plot(self.ax, self.x, dimensions[0], dimensions[0], c='b')
        Axes3D.plot(self.ax, self.x, dimensions[2], dimensions[0], c='b')
        Axes3D.plot(self.ax, self.x, dimensions[0], dimensions[3], c='b')
        Axes3D.plot(self.ax, self.x, dimensions[2], dimensions[3], c='b')

        Axes3D.plot(self.ax, dimensions[0], self.y, dimensions[0], c='b')
        Axes3D.plot(self.ax, dimensions[0], self.y, dimensions[3], c='b')
        Axes3D.plot(self.ax, dimensions[1], self.y, dimensions[0], c='b')
        Axes3D.plot(self.ax, dimensions[1], self.y, dimensions[3], c='b')

        Axes3D.plot(self.ax, dimensions[0], dimensions[0], self.z, c='b')
        Axes3D.plot(self.ax, dimensions[0], dimensions[2], self.z, c='b')
        Axes3D.plot(self.ax, dimensions[1], dimensions[0], self.z, c='b')
        Axes3D.plot(self.ax, dimensions[1], dimensions[2], self.z, c='b')

        # Draw nodes if shown:
        if is_sdn_shown:
            for node in self.nodes:
                Axes3D.scatter(ax, node.x, node.y, node.z, color='g')

        # Draw images if shown:
        if is_images_shown:
            Axes3D.scatter(ax, -source.x, source.y, source.z, color='y')
            Axes3D.scatter(ax, 2 * self.x[1] - source.x, source.y, source.z, color='y')

            Axes3D.scatter(ax, source.x, -source.y, source.z, color='y')
            Axes3D.scatter(ax, source.x, 2 * self.y[1] - source.y, source.z, color='y')

            Axes3D.scatter(ax, source.x, source.y, -source.z, color='y')
            Axes3D.scatter(ax, source.x, source.y, 2 * self.z[1] - source.z, color='y')
        plt.show()

    def create_delay_lines(self):
        # Delay line between microphone and node
        source = self.source
        microphone = self.microphone

        node_list = self.nodes

        delay_lines = []

        # Outgoing_delay_line from source to microphone
        delay_line = DelayLine(source, microphone)
        delay_lines.append(delay_line)
        source.outgoing_delay_lines.append(delay_line)

        # Outgoing_delay_lines from source to microphone
        for node in node_list:
            delay_line = DelayLine(source, node)
            delay_lines.append(delay_line)
            source.outgoing_delay_lines.append(delay_line)

        # Creation of outgoing_delay_lines between nodes
        for i in range(0, len(node_list)):
            delay_line = DelayLine(node_list[i], microphone)
            delay_lines.append(delay_line)
            for j in range(0, len(node_list)):
                if i == j:
                    continue
                else:
                    delay_line = DelayLine(node_list[i], node_list[j])
                    delay_lines.append(delay_line)
                    node_list[i].outgoing_delay_lines.append(delay_line)
        # Creation of incoming_delay_lines between nodes
        for node in node_list:
            for delay_line in delay_lines:
                if delay_line.start is source:
                    continue
                if delay_line.end is node:
                    node.incoming_delay_lines.append(delay_line)

        # Incoming_delay_lines to microphone
        for delay_line in delay_lines:
            if delay_line.end is microphone:
                microphone.incoming_delay_lines.append(delay_line)

        return delay_lines


    def find_distances(self):
        source = self.source
        microphone = self.microphone

        for node in self.nodes:
            source.distance_values[node.index] = np.linalg.norm(source.position - node.position)
            microphone.distance_values[node.index] = np.linalg.norm(microphone.position - node.position)
        source.distance_values[0] = np.linalg.norm(source.position - microphone.position)
        microphone.distance_values[0] = np.linalg.norm(microphone.position - source.position)

    def TickFunction(self):
        source = self.source
        microphone = self.microphone
        # For source:
        for outgoing_delay_line in source.outgoing_delay_lines:
            if outgoing_delay_line.end is not microphone:
                outgoing_delay_line.buffer.appendleft(source.input[-1] *
                                                      0.5 *
                                                      1 / (source.distance_values[outgoing_delay_line.end.index]))
                for incoming_delay_line in outgoing_delay_line.end.incoming_delay_lines:
                    incoming_delay_line.buffer[-1] += outgoing_delay_line.buffer[-1]
                outgoing_delay_line.buffer.pop()
            else:
                outgoing_delay_line.buffer.appendleft(source.input[-1] *
                                                    1 / (source.distance_values[0]))
        source.input.appendleft(0.0)
        source.input.pop()
        # For scattering nodes:
        for node in self.nodes:
            for i in range(0, len(node.outgoing_delay_lines)):
                incoming_sum = 0.0
                for j in range(0, len(node.incoming_delay_lines)):
                    incoming_sum += SCATTERING_MATRIX[i][j] * \
                                    node.incoming_delay_lines[j].buffer[-1]
                incoming_sum = incoming_sum * self.WALL_REFLECTION_COEFFICIENT
                node.outgoing_delay_lines[i].buffer.appendleft(incoming_sum)
            for delay_line in node.incoming_delay_lines:
                delay_line.buffer.pop()
        # For microphone:
        for incoming_delay_line in microphone.incoming_delay_lines:
            if incoming_delay_line.start is not source:
                outgoing_sum = 0.0
                for outgoing_delay_line in incoming_delay_line.start.outgoing_delay_lines:
                    outgoing_sum += outgoing_delay_line.buffer[0]
                outgoing_sum = outgoing_sum * \
                               (2 / 5) * self.WALL_REFLECTION_COEFFICIENT * \
                               1 / (1 + (microphone.distance_values[incoming_delay_line.start.index] /
                                    source.distance_values[incoming_delay_line.start.index]))
                incoming_delay_line.buffer.appendleft(outgoing_sum)
        output_sum = 0.0
        for incoming_delay_line in microphone.incoming_delay_lines:
            output_sum += incoming_delay_line.buffer.pop()
        microphone.output.append(output_sum)


class Microphone(object):
    def __init__(self, position):
        self.label = "Microphone"

        self.position = position
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]

        self.output = col.deque([])

        self.incoming_delay_lines = []


        self.distance_values = [None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None]

    def __str__(self):
        return self.label

    def __repr__(self):
        return str(self)


class Source(object):
    def __init__(self, position):
        self.label = "Source"

        self.position = position
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]


        self.outgoing_delay_lines = []

        self.input = col.deque([])

        self.distance_values = [None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None]

    def __repr__(self):
        return self.label


class SDN_Node(object):
    def __init__(self, x, y, z, label):
        self.x = x
        self.y = y
        self.z = z

        self.position = np.array([self.x, self.y, self.z])
        self.label = label

        self.index = 0

        self.incoming_delay_lines = []

        self.outgoing_delay_lines = []

    def __repr__(self):
        return self.label


class DelayLine(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

        self.label = str(start) + " to " + str(end)

        self.distance = np.linalg.norm(start.position - end.position)

        length = int((SAMPLING_RATE * self.distance) / SPEED_OF_SOUND)

        self.length = length

        self.buffer = col.deque(np.zeros(self.length, dtype= float))

    def __repr__(self):
        return self.label


class SoundFileRW(object):
    def __init__(self, rate, data):
        self.data = None
        self.rate = None

    def read_sound_file(self, filename, source):
        # If you want to feed a wav file:
        # self.rate, self.data = wavfile.read(filename)
        # self.data.flags. writeable = True
        # self.data = self.data.astype(float)

        # If you want to feed an impulse:
        self.rate, self.data = wavfile.read(filename)
        self.data = sig.unit_impulse(1)

        source.input = col.deque(np.flip(self.data))

    def write_sound_file(self, filename, microphone):
        self.data = np.array(microphone.output)
        self.data = self.data / np.max(np.abs(self.data))
        self.data = np.trim_zeros(self.data, 'f')

        time_passed_one = np.arange(0, len(self.data) / SAMPLING_RATE, (1 / SAMPLING_RATE))


        # signal_ned = np.zeros(len(self.data))

        # window_size = round(44100 * 0.01)
        # window = np.hanning(window_size)
        # window = window / sum(window)
        # window_length = len(window)


        # for i in range(0, len(self.data) - window_length):
        #     values = self.data[i:(i + window_length)]
        #     sigma = math.sqrt(sum(np.multiply(window, np.square(values))))
        #
        #     sumValues = 0
        #     for value in values:
        #         if abs(value) > sigma:
        #             sumValues += window[values.tolist().index(value)]
        #
        #     signal_ned[i] = (1 / math.erfc(1 / math.sqrt(2))) * sumValues

        plt.figure()
        plt.axis([-0.005, 0.06, -0.1, 1.2])
        plt.plot(time_passed_one, self.data)
        # plt.plot(time_passed_one,signal_ned)
        plt.show()

        Fs = self.rate
        Fs = float(Fs)
        Lp = len(self.data)
        Tp = np.arange(0, Lp / Fs, (1 / Fs))

        pEnergy = (np.cumsum(self.data[::-1] ** 2) / np.sum(self.data[::-1]))[::-1]
        pEdB = 10.0 * np.log10(pEnergy / np.max(pEnergy))
        Tp = Tp[:len(pEdB)]  # Trim Tp

        plt.figure()
        plt.plot(Tp, pEdB, ls = 'solid', color = 'b', label= 'Energy decay curve', linewidth = 1)
        plt.show()
        self.data = np.asarray(microphone.output, dtype=np.int16)
        # wavfile.write(filename, self.rate, self.data)


if __name__ == "__main__":
    t0 = time.time()
    # Set up microphone
    mic = Microphone(np.array([2, 2, 1.5]))
    # Set up source
    src = Source(np.array([4.5, 3.5, 2]))
    # Set up Room
    shoebox_room = Room(9, 7, 4,
                        0.2,
                        mic,
                        src)

    # Find images
    shoebox_room.find_images()

    # Find sdn nodes
    shoebox_room.find_sdn_nodes()


    # Plot Room
    shoebox_room.Plot(True, True)

    # Create Delay Lines:
    delayLines = shoebox_room.create_delay_lines()

    print(delayLines)
    print(len(delayLines))

    # Find distances:
    shoebox_room.find_distances()

    # Read File and input to source
    read_write = SoundFileRW(0, 0)
    read_write.read_sound_file(r"./samples/mozart_bsn_5.wav", src)


    t1 = time.time()
    print("Generation is finished in: ", t1 - t0, " seconds!")
    upto = SAMPLING_RATE//5
    for i in range(0, upto):
        shoebox_room.TickFunction()
    t2 = time.time()
    print("Calculation is finished in: ", t2 - t1, " seconds!")

    read_write.write_sound_file("impulse.wav", mic)


# total delay line lengths
# s = 0
# for i in mic.incoming_delay_lines:
#     s += len(i.buffer)
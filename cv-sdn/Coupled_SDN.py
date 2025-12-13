"""
Timuçin Berk Atalay
ID:1736214
Multimedia Informatics - Thesis Work
Coupled-SDN Thesis Work
"""

# Python modules
import random
import math
import collections

# Numpy-Scipy-Matplotlib modules
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import scipy.signal as sig

# Constants:
SCATTERING_MATRIX = (2 / 5 * np.ones((5, 5)) - np.identity(5))
SAMPLING_RATE = 44100
SPEED_OF_SOUND = 343


def line_intersection(image, target_object, point1, point2, point3):
    vector1 = point3 - point1
    vector2 = point2 - point1

    plane_normal = np.cross(vector1, vector2)

    # Line point and direction
    line_direction = np.array([image[0] - target_object.x, image[1] - target_object.y, image[2] - target_object.z])

    line_point = np.array([target_object.x, target_object.y, target_object.z])

    # Calculation of intersection
    normal_vector = plane_normal.dot(line_direction)
    w = line_point - point1
    si = -plane_normal.dot(w) / normal_vector
    intersection_point = w + si * line_direction + point1

    return intersection_point


def directivity_calculator(point, aperture, mode=0):
    # Mode = 0 corresponds to omnidirectional aperture
    if mode == 0:
        return 1
    # Mode = 1 corresponds to bidirectional microphone
    elif mode == 1:
        point1 = np.array([aperture.x, random.uniform(0, aperture.y), random.uniform(0, aperture.z)])
        point2 = np.array([aperture.x, random.uniform(0, aperture.y), random.uniform(0, aperture.z)])
        point3 = np.array([aperture.x, random.uniform(0, aperture.y), random.uniform(0, aperture.z)])

        vector1 = point3 - point1
        vector2 = point2 - point1

        normal_vector = np.cross(vector1, vector2)

        unit_vector = normal_vector / np.linalg.norm(normal_vector)

        directional_vector = aperture.position - point.position

        unit_directional = directional_vector / np.linalg.norm(directional_vector)


        dot_product = np.dot(unit_vector, unit_directional)

        angle = np.arccos(math.fabs(dot_product))

        # print(angle * 180 / math.pi)
        # print(math.cos(angle))
        return math.cos(angle)
    else:
        return 1


class Room(object):
    """
    Creates the first room with the given parameters:
    1- Absorption Coefficient
    2- Microphone
    3- Source
    4- Door

    First, the wall reflection coefficient is calculated from the absorption coefficient.
    Secondly, Room coordinates are created with numpy arrays.
    Thirdly, image list, node list, and second_room is initialized as None.
    Finally, positions of the microphone, source, and the door are checked.
    If placed correctly, the room object is created.

    If microphone and/or source does not reside in the room, insert None to create the room.

    """

    def __init__(self, x, y, z
                 , absorption_coefficient
                 , microphone
                 , source
                 , door
                 , directivity_mode):

        self.WALL_REFLECTION_COEFFICIENT = -math.sqrt(1 - absorption_coefficient)
        self.absorption_coefficient = absorption_coefficient

        self.x = np.array([0., x])
        self.y = np.array([0., y])
        self.z = np.array([0., z])

        self.images = []
        self.door_images = []

        self.nodes = []
        self.doormic_nodes = []
        self.doorsrc_nodes = []
        self.doordoor_nodes = []

        self.directivity_mode = directivity_mode

        self.second_room = None

        if microphone is not None:
            if microphone.x > x or microphone.y > y or microphone.z > z \
                    or microphone.x < 0 or microphone.y < 0 or microphone.z < 0:
                raise AssertionError("Microphone is out of bounds!")

        if source is not None:
            if source.x > x or source.y > y or source.z > z \
                    or source.x < 0 or source.y < 0 or source.z < 0:
                raise AssertionError("Source is out of bounds!")

        if door.x != self.x[1]:
            raise AssertionError("Door is not placed correctly!")

        self.microphone = microphone
        self.source = source
        self.door = door

        self.is_microphone_in_room = microphone is not None
        self.is_source_in_room = source is not None

        self.total_area = self.y[1] * self.z[1]

        self.total_area = self.y[1] * self.z[1]
        self.common_wall_absorption = (self.door.area + (
                self.total_area - door.area) * self.absorption_coefficient) / self.total_area

        print(self.total_area)
        self.max_area = 2 * self.y[1] * self.z[1] + 2 * self.x[1] * self.z[1] + 2 * self.y[1] * self.x[1]

        print(self.max_area)

    def find_images(self):
        """
        Image method finds the images in the room with respect to source and microphone existence.

        If the source is in the room, source's images are found no matter what.
        If the microphone is in the room, door's images are found.
        If both source and microphone are missing, there is no need to find images as scattering nodes of an empty room
        will be placed at the center of walls.
        """

        source = self.source
        door = self.door

        if self.is_source_in_room and self.is_microphone_in_room:

            self.images.append(np.array([-source.x, source.y, source.z]))
            self.images.append(np.array([2 * self.x[1] - source.x, source.y, source.z]))

            self.images.append(np.array([source.x, -source.y, source.z]))
            self.images.append(np.array([source.x, 2 * self.y[1] - source.y, source.z]))

            self.images.append(np.array([source.x, source.y, -source.z]))
            self.images.append(np.array([source.x, source.y, 2 * self.z[1] - source.z]))

            self.door_images.append(np.array([-door.x, door.y, door.z]))

            self.door_images.append(np.array([door.x, -door.y, door.z]))
            self.door_images.append(np.array([door.x, 2 * self.y[1] - door.y, door.z]))

            self.door_images.append(np.array([door.x, door.y, -door.z]))
            self.door_images.append(np.array([door.x, door.y, 2 * self.z[1] - door.z]))


        elif self.is_source_in_room and not self.is_microphone_in_room:

            self.images.append(np.array([-source.x, source.y, source.z]))
            self.images.append(np.array([2 * self.x[1] - source.x, source.y, source.z]))

            self.images.append(np.array([source.x, -source.y, source.z]))
            self.images.append(np.array([source.x, 2 * self.y[1] - source.y, source.z]))

            self.images.append(np.array([source.x, source.y, -source.z]))
            self.images.append(np.array([source.x, source.y, 2 * self.z[1] - source.z]))

        elif not self.is_source_in_room and self.is_microphone_in_room:

            self.images.append(np.array([-door.x, door.y, door.z]))

            self.images.append(np.array([door.x, -door.y, door.z]))
            self.images.append(np.array([door.x, 2 * self.y[1] - door.y, door.z]))

            self.images.append(np.array([door.x, door.y, -door.z]))
            self.images.append(np.array([door.x, door.y, 2 * self.z[1] - door.z]))

    def find_sdn_nodes(self):

        microphone = self.microphone
        door = self.door

        if self.is_source_in_room and self.is_microphone_in_room:

            # door-src nodes in first room

            self.nodes = self.sdn_finder(microphone, self.images, self.nodes)

            # door-mic nodes in first room

            new_node = ScatteringNode(np.array([self.x[1], self.y[1] / 2, self.z[1] / 2]), 3)
            self.doormic_nodes.append(new_node)
            self.doormic_nodes = self.sdn_finder(microphone, self.door_images, self.doormic_nodes)

            # door-src nodes in first room

            self.doorsrc_nodes = self.sdn_finder(door, self.images, self.doorsrc_nodes)

            # door-door nodes in first room

            new_node = ScatteringNode(np.array([self.x[0], (self.y[1] - self.y[0]) / 2, (self.z[1] - self.z[0]) / 2]),
                                      1)
            self.doordoor_nodes.append(new_node)

            new_node = ScatteringNode(
                np.array([(self.x[1] - self.x[0]) / 2 + self.x[0], self.y[1], (self.z[1] - self.z[0]) / 2]), 2)
            self.doordoor_nodes.append(new_node)

            new_node = ScatteringNode(np.array([self.x[1], (self.y[1] - self.y[0]) / 2, (self.z[1] - self.z[0]) / 2]),
                                      3)
            self.doordoor_nodes.append(new_node)

            new_node = ScatteringNode(
                np.array([(self.x[1] - self.x[0]) / 2 + self.x[0], self.y[0], (self.z[1] - self.z[0]) / 2]), 4)
            self.doordoor_nodes.append(new_node)

            new_node = ScatteringNode(
                np.array([(self.x[1] - self.x[0]) / 2 + self.x[0], (self.y[1] - self.y[0]) / 2, self.z[0]]),
                5)
            self.doordoor_nodes.append(new_node)

            new_node = ScatteringNode(
                np.array([(self.x[1] - self.x[0]) / 2 + self.x[0], (self.y[1] - self.y[0]) / 2, self.z[1]]),
                6)
            self.doordoor_nodes.append(new_node)

        elif not self.is_source_in_room and not self.is_microphone_in_room:
            new_node = ScatteringNode(np.array([self.x[0], (self.y[1] - self.y[0]) / 2, (self.z[1] - self.z[0]) / 2]),
                                      1)
            self.nodes.append(new_node)

            new_node = ScatteringNode(
                np.array([(self.x[1] - self.x[0]) / 2 + self.x[0], self.y[1], (self.z[1] - self.z[0]) / 2]), 2)
            self.nodes.append(new_node)

            new_node = ScatteringNode(np.array([self.x[1], (self.y[1] - self.y[0]) / 2, (self.z[1] - self.z[0]) / 2]),
                                      3)
            self.nodes.append(new_node)

            new_node = ScatteringNode(
                np.array([(self.x[1] - self.x[0]) / 2 + self.x[0], self.y[0], (self.z[1] - self.z[0]) / 2]), 4)
            self.nodes.append(new_node)

            new_node = ScatteringNode(
                np.array([(self.x[1] - self.x[0]) / 2 + self.x[0], (self.y[1] - self.y[0]) / 2, self.z[0]]),
                5)
            self.nodes.append(new_node)

            new_node = ScatteringNode(
                np.array([(self.x[1] - self.x[0]) / 2 + self.x[0], (self.y[1] - self.y[0]) / 2, self.z[1]]),
                6)
            self.nodes.append(new_node)

        elif self.is_source_in_room and not self.is_microphone_in_room:

            new_node = ScatteringNode(np.array([self.x[1], self.y[1] / 2, self.z[1] / 2]), 3)
            self.nodes.append(new_node)
            self.nodes = self.sdn_finder(door, self.images, self.nodes)


        else:

            new_node = ScatteringNode(np.array([self.x[1], self.y[1] / 2, self.z[1] / 2]), 3)
            self.nodes.append(new_node)

            self.nodes = self.sdn_finder(microphone, self.images, self.nodes)

    def sdn_finder(self, target_object, image_list, node_list):

        x = self.x[1]
        y = self.y[1]
        z = self.z[1]

        images = image_list
        nodes = node_list

        for image in images:
            # to-do: Find a simpler way of performing SDN_Calculation
            if image[0] < 0:

                # Plane points and normal vector
                point1 = np.array([0, random.uniform(0, y), random.uniform(0, z)])
                point2 = np.array([0, random.uniform(0, y), random.uniform(0, z)])
                point3 = np.array([0, random.uniform(0, y), random.uniform(0, z)])

                intersection_point = line_intersection(image, target_object, point1, point2, point3)

                new_node = ScatteringNode(intersection_point, 1)
                nodes.append(new_node)

            elif image[1] > y:
                # Plane points and normal vector
                point1 = np.array([random.uniform(0, x), y, random.uniform(0, z)])
                point2 = np.array([random.uniform(0, x), y, random.uniform(0, z)])
                point3 = np.array([random.uniform(0, x), y, random.uniform(0, z)])

                intersection_point = line_intersection(image, target_object, point1, point2, point3)

                new_node = ScatteringNode(intersection_point, 2)
                nodes.append(new_node)

            elif image[0] > x:
                # Plane points and normal vector
                point1 = np.array([x, random.uniform(0, y), random.uniform(0, z)])
                point2 = np.array([x, random.uniform(0, y), random.uniform(0, z)])
                point3 = np.array([x, random.uniform(0, y), random.uniform(0, z)])

                intersection_point = line_intersection(image, target_object, point1, point2, point3)

                new_node = ScatteringNode(intersection_point, 3)

                is_already_available = False
                for node in nodes:
                    if node.index == 3:
                        is_already_available = True
                if not is_already_available:
                    nodes.append(new_node)

            elif image[1] < 0:
                # Plane points and normal vector
                point1 = np.array([random.uniform(0, x), 0, random.uniform(0, z)])
                point2 = np.array([random.uniform(0, x), 0, random.uniform(0, z)])
                point3 = np.array([random.uniform(0, x), 0, random.uniform(0, z)])

                intersection_point = line_intersection(image, target_object, point1, point2, point3)

                new_node = ScatteringNode(intersection_point, 4)
                nodes.append(new_node)

            elif image[2] < 0:

                # Plane points and normal vector
                point1 = np.array([random.uniform(0, x), random.uniform(0, y), 0])
                point2 = np.array([random.uniform(0, x), random.uniform(0, y), 0])
                point3 = np.array([random.uniform(0, x), random.uniform(0, y), 0])

                intersection_point = line_intersection(image, target_object, point1, point2, point3)

                new_node = ScatteringNode(intersection_point, 5)
                nodes.append(new_node)

            elif image[2] > z:
                # Plane points and normal vector
                point1 = np.array([random.uniform(0, x), random.uniform(0, y), z])
                point2 = np.array([random.uniform(0, x), random.uniform(0, y), z])
                point3 = np.array([random.uniform(0, x), random.uniform(0, y), z])

                intersection_point = line_intersection(image, target_object, point1, point2, point3)

                new_node = ScatteringNode(intersection_point, 6)
                nodes.append(new_node)

            else:
                pass

        # Order SDN nodes by their indexes
        new_nodes = []
        for node in nodes:
            new_nodes.insert(node.index - 1, node)
        nodes = new_nodes

        return nodes

    def create_delay_lines(self):

        delay_lines = []

        source = self.source
        microphone = self.microphone
        door = self.door

        # For 2 SDN enable this part and disable for 5 SDN:
        for i in range(0, 6):
            self.doormic_nodes[i].position = self.nodes[i].position
            self.doorsrc_nodes[i].position = self.nodes[i].position
            self.doordoor_nodes[i].position = self.nodes[i].position

        # For classic 5 SDN mode use these
        node_list = self.nodes
        doormic_node_list = self.doormic_nodes
        doorsrc_node_list = self.doorsrc_nodes
        doordoor_node_list = self.doordoor_nodes

        if self.is_source_in_room and self.is_microphone_in_room:

            # WASPAA için eklenen kısım:
            # Outgoing_delay_line from door to nodes
            for node in doordoor_node_list:
                if node.index % 10 == 3:
                    continue
                delay_line = DelayLine(door, node)
                delay_lines.append(delay_line)
                door.outgoing_delay_lines.append(delay_line)
            # Creation of outgoing_delay_lines between nodes
            for i in range(0, len(doordoor_node_list)):
                delay_line = DelayLine(doordoor_node_list[i], door)
                delay_lines.append(delay_line)
                if i != 2:
                    door.incoming_delay_lines.append(delay_line)
                for j in range(0, len(doordoor_node_list)):
                    if i == j:
                        continue
                    else:
                        delay_line = DelayLine(doordoor_node_list[i], doordoor_node_list[j])
                        delay_lines.append(delay_line)
                        doordoor_node_list[i].outgoing_delay_lines.append(delay_line)
            # Creation of incoming_delay_lines between nodes
            for node in doordoor_node_list:
                for delay_line in delay_lines:
                    if delay_line.start is door:
                        continue
                    if delay_line.end is node:
                        node.incoming_delay_lines.append(delay_line)
            # Index numbering:
            for node in doordoor_node_list:
                node.index = node.index + 30

            # Outgoing_delay_line from source to microphone
            delay_line = DelayLine(source, microphone)
            delay_lines.append(delay_line)
            source.outgoing_delay_lines.append(delay_line)

            # Outgoing_delay_line from source to door
            delay_line = DelayLine(source, door)
            delay_lines.append(delay_line)
            source.outgoing_delay_lines.append(delay_line)
            door.incoming_delay_lines.append(delay_line)

            # Outgoing_delay_lines from source to original nodes
            for node in node_list:
                delay_line = DelayLine(source, node)
                delay_lines.append(delay_line)
                source.outgoing_delay_lines.append(delay_line)

            # Outgoing_delay_lines from source to door nodes
            for node in doorsrc_node_list:
                node.index = node.index + 20
                delay_line = DelayLine(source, node)
                delay_lines.append(delay_line)
                source.outgoing_delay_lines.append(delay_line)

            # Incoming_delay_lines from source nodes to door
            for node in doorsrc_node_list:
                if node.index == 23:
                    continue
                delay_line = DelayLine(node, door)
                delay_lines.append(delay_line)
                door.incoming_delay_lines.append(delay_line)

            # Outgoing_delay_lines between source-mic nodes
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

            # Incoming_delay_lines between source-mic nodes
            for node in node_list:
                for delay_line in delay_lines:
                    if delay_line.start is source:
                        continue
                    if delay_line.end is node:
                        node.incoming_delay_lines.append(delay_line)

            # Outgoing_delay_lines between source-door nodes
            for i in range(0, len(doorsrc_node_list)):
                delay_line = DelayLine(doorsrc_node_list[i], door)
                delay_lines.append(delay_line)
                for j in range(0, len(doorsrc_node_list)):
                    if i == j:
                        continue
                    else:
                        delay_line = DelayLine(doorsrc_node_list[i], doorsrc_node_list[j])
                        delay_lines.append(delay_line)
                        doorsrc_node_list[i].outgoing_delay_lines.append(delay_line)

            # Incoming_delay_lines between source-door nodes
            for node in doorsrc_node_list:
                for delay_line in delay_lines:
                    if delay_line.start is source:
                        continue
                    if delay_line.end is node:
                        node.incoming_delay_lines.append(delay_line)

            # Outgoing_delay_line from door to microphone
            delay_line = DelayLine(door, microphone)
            delay_lines.append(delay_line)
            door.outgoing_delay_lines.append(delay_line)

            # Outgoing_delay_lines from door to doormic_nodes
            for node in doormic_node_list:
                node.index = node.index + 10
                if node.index == 13:
                    continue
                delay_line = DelayLine(door, node)
                delay_lines.append(delay_line)
                door.outgoing_delay_lines.append(delay_line)

            # Outgoing_delay_lines between doormic_nodes
            for i in range(0, len(doormic_node_list)):
                delay_line = DelayLine(doormic_node_list[i], microphone)
                delay_lines.append(delay_line)
                for j in range(0, len(doormic_node_list)):
                    if i == j:
                        continue
                    else:
                        delay_line = DelayLine(doormic_node_list[i], doormic_node_list[j])
                        delay_lines.append(delay_line)
                        doormic_node_list[i].outgoing_delay_lines.append(delay_line)

            # Incoming_delay_lines between doormic_nodes
            for node in doormic_node_list:
                for delay_line in delay_lines:
                    if delay_line.start is door:
                        continue
                    if delay_line.end is node:
                        node.incoming_delay_lines.append(delay_line)

            # Incoming_delay_lines to microphone
            for delay_line in delay_lines:
                if delay_line.end is microphone:
                    microphone.incoming_delay_lines.append(delay_line)
        elif self.is_source_in_room and not self.is_microphone_in_room:

            # Outgoing_delay_line from source to door
            delay_line = DelayLine(source, door)
            delay_lines.append(delay_line)
            source.outgoing_delay_lines.append(delay_line)

            # Outgoing_delay_lines from source to door
            for node in node_list:
                delay_line = DelayLine(source, node)
                delay_lines.append(delay_line)
                source.outgoing_delay_lines.append(delay_line)
            # Creation of incoming_delay_lines between nodes
            for i in range(0, len(node_list)):
                delay_line = DelayLine(node_list[i], door)
                delay_lines.append(delay_line)
                for j in range(0, len(node_list)):
                    if i == j:
                        continue
                    else:
                        delay_line = DelayLine(node_list[i], node_list[j])
                        delay_lines.append(delay_line)
                        node_list[i].outgoing_delay_lines.append(delay_line)
            for node in node_list:
                for delay_line in delay_lines:
                    if delay_line.start is source:
                        continue
                    if delay_line.end is node:
                        node.incoming_delay_lines.append(delay_line)
            # Incoming_delay_lines to microphone
            for delay_line in delay_lines:
                if delay_line.end is door:
                    door.incoming_delay_lines.append(delay_line)
        elif not self.is_source_in_room and self.is_microphone_in_room:

            # Outgoing_delay_line from source to door
            delay_line = DelayLine(door, microphone)
            delay_lines.append(delay_line)
            door.outgoing_delay_lines.append(delay_line)

            # Outgoing_delay_lines from source to door
            for node in node_list:
                delay_line = DelayLine(door, node)
                delay_lines.append(delay_line)
                door.outgoing_delay_lines.append(delay_line)
            # Creation of incoming_delay_lines between nodes
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
            for node in node_list:
                for delay_line in delay_lines:
                    if delay_line.start is door:
                        continue
                    if delay_line.end is node:
                        node.incoming_delay_lines.append(delay_line)
            # Incoming_delay_lines to microphone
            for delay_line in delay_lines:
                if delay_line.end is microphone:
                    microphone.incoming_delay_lines.append(delay_line)
        else:

            # Outgoing_delay_line from door to nodes
            for node in node_list:
                if node.index == 3:
                    continue
                delay_line = DelayLine(door, node)
                delay_lines.append(delay_line)
                door.outgoing_delay_lines.append(delay_line)

            # Creation of outgoing_delay_lines between nodes
            for i in range(0, len(node_list)):
                delay_line = DelayLine(node_list[i], door)
                delay_lines.append(delay_line)
                if i != 0:
                    door.incoming_delay_lines.append(delay_line)
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
                    if delay_line.start is door:
                        continue
                    if delay_line.end is node:
                        node.incoming_delay_lines.append(delay_line)

        return delay_lines

    def find_distances(self):

        source = self.source
        microphone = self.microphone
        door = self.door

        if self.is_source_in_room and self.is_microphone_in_room:

            for node in self.nodes:
                source.distance_values[node.index] = np.linalg.norm(source.position - node.position)
                microphone.distance_values[node.index] = np.linalg.norm(microphone.position - node.position)
            for node in self.doormic_nodes:
                door.first_room_distance_values[node.index - 10] = np.linalg.norm(node.position - door.position)
                microphone.door_distance_values[node.index - 10] = np.linalg.norm(node.position - microphone.position)
            for node in self.doorsrc_nodes:
                source.door_distance_values[node.index - 20] = np.linalg.norm(node.position - source.position)
                door.first_room_source_distance_values[node.index - 20] = np.linalg.norm(node.position - door.position)
            for node in self.doordoor_nodes:
                door.first_room_door_distance_values[node.index - 30] = np.linalg.norm(door.position - node.position)

            source.distance_values[0] = np.linalg.norm(source.position - microphone.position)
            source.door_distance_values[0] = np.linalg.norm(source.position - door.position)
            microphone.distance_values[0] = np.linalg.norm(microphone.position - source.position)
            microphone.door_distance_values[0] = np.linalg.norm(microphone.position - door.position)
            door.first_room_distance_values[0] = np.linalg.norm(door.position - microphone.position)
            door.first_room_source_distance_values[0] = np.linalg.norm(door.position - source.position)

        elif self.is_source_in_room and not self.is_microphone_in_room:

            for node in self.nodes:
                source.distance_values[node.index] = np.linalg.norm(source.position - node.position)
                door.first_room_distance_values[node.index] = np.linalg.norm(door.position - node.position)
            source.distance_values[0] = np.linalg.norm(source.position - door.position)
            door.first_room_distance_values[0] = np.linalg.norm(door.position - source.position)

        elif not self.is_source_in_room and self.is_microphone_in_room:

            for node in self.nodes:
                microphone.distance_values[node.index] = np.linalg.norm(microphone.position - node.position)
                door.first_room_distance_values[node.index] = np.linalg.norm(door.position - node.position)
                microphone.distance_values[0] = np.linalg.norm(microphone.position - door.position)
            door.first_room_distance_values[0] = np.linalg.norm(door.position - microphone.position)

        else:

            for node in self.nodes:
                door.first_room_distance_values[node.index] = np.linalg.norm(door.position - node.position)

    def tick_function(self):

        source = self.source
        microphone = self.microphone
        door = self.door
        second_room = self.second_room

        if self.is_source_in_room and self.is_microphone_in_room:
            # For source:
            for outgoing_delay_line in source.outgoing_delay_lines:
                if outgoing_delay_line.end is not microphone and outgoing_delay_line.end is not door:
                    if outgoing_delay_line.end.index < 10:
                        if outgoing_delay_line.end.index == 3:
                            outgoing_delay_line.buffer.appendleft(source.input[-1] *
                                                                  0.5 *
                                                                  1 / (
                                                                      source.distance_values[
                                                                          outgoing_delay_line.end.index]))
                        else:
                            outgoing_delay_line.buffer.appendleft(source.input[-1] *
                                                                  0.5 *
                                                                  1 / (
                                                                      source.distance_values[
                                                                          outgoing_delay_line.end.index]))
                        for incoming_delay_line in outgoing_delay_line.end.incoming_delay_lines:
                            incoming_delay_line.buffer[-1] += outgoing_delay_line.buffer[-1]
                        outgoing_delay_line.buffer.pop()
                    elif 20 < outgoing_delay_line.end.index < 30:
                        if outgoing_delay_line.end.index % 10 == 3:
                            outgoing_delay_line.buffer.appendleft(source.input[-1] *
                                                                  0.5 *
                                                                  1 / (source.door_distance_values[
                                outgoing_delay_line.end.index - 20]))
                        else:
                            outgoing_delay_line.buffer.appendleft(source.input[-1] *
                                                                  0.5 *
                                                                  1 / (source.door_distance_values[
                                outgoing_delay_line.end.index - 20]))
                        for incoming_delay_line in outgoing_delay_line.end.incoming_delay_lines:
                            incoming_delay_line.buffer[-1] += outgoing_delay_line.buffer[-1]
                        outgoing_delay_line.buffer.pop()
                elif outgoing_delay_line.end is microphone:
                    outgoing_delay_line.buffer.appendleft(source.input[-1] *
                                                          1 / (source.distance_values[0]))
                elif outgoing_delay_line.end is door:
                    outgoing_delay_line.buffer.appendleft(source.input[-1]  *
                                                          1 / (source.door_distance_values[0]))
            source.input.appendleft(0.0)
            source.input.pop()
            # For scattering nodes in first room (source-mic side):
            for node in self.nodes:
                for i in range(0, len(node.outgoing_delay_lines)):
                    incoming_sum = 0.0
                    for j in range(0, len(node.incoming_delay_lines)):
                        incoming_sum += SCATTERING_MATRIX[i][j] * \
                                        node.incoming_delay_lines[j].buffer[-1]
                    if node.index % 10 == 3:
                        incoming_sum = incoming_sum * math.sqrt(1 - self.common_wall_absorption)
                    else:
                        incoming_sum = incoming_sum * self.WALL_REFLECTION_COEFFICIENT
                    node.outgoing_delay_lines[i].buffer.appendleft(incoming_sum)
                for delay_line in node.incoming_delay_lines:
                    delay_line.buffer.pop()
            # For scattering nodes in first room (source-door side):
            for node in self.doorsrc_nodes:
                for i in range(0, len(node.outgoing_delay_lines)):
                    incoming_sum = 0.0
                    for j in range(0, len(node.incoming_delay_lines)):
                        incoming_sum += SCATTERING_MATRIX[i][j] * \
                                        node.incoming_delay_lines[j].buffer[-1]
                    if node.index % 10 == 3:
                        incoming_sum = incoming_sum * math.sqrt(1 - self.common_wall_absorption)
                    else:
                        incoming_sum = incoming_sum * self.WALL_REFLECTION_COEFFICIENT
                    node.outgoing_delay_lines[i].buffer.appendleft(incoming_sum)
                for delay_line in node.incoming_delay_lines:
                    delay_line.buffer.pop()
            # For door as microphone in the first room:
            for incoming_delay_line in door.incoming_delay_lines:
                if incoming_delay_line.start is not source and 20 < incoming_delay_line.start.index < 30:
                    outgoing_sum = 0.0
                    for outgoing_delay_line in incoming_delay_line.start.outgoing_delay_lines:
                        outgoing_sum += outgoing_delay_line.buffer[0]
                    outgoing_sum = outgoing_sum * \
                                   (2 / 5) * self.WALL_REFLECTION_COEFFICIENT * \
                                   1 / (1 + (
                            door.first_room_source_distance_values[incoming_delay_line.start.index - 20] /
                            source.door_distance_values[incoming_delay_line.start.index - 20]))
                    incoming_delay_line.buffer.appendleft(outgoing_sum * directivity_calculator(incoming_delay_line.start, door, self.directivity_mode))
                elif incoming_delay_line.start is not source and incoming_delay_line.start.index > 30:
                    outgoing_sum = 0.0
                    for outgoing_delay_line in incoming_delay_line.start.outgoing_delay_lines:
                        outgoing_sum += outgoing_delay_line.buffer[0]
                    outgoing_sum = outgoing_sum * \
                                   (2 / 5) * self.WALL_REFLECTION_COEFFICIENT * \
                                   1 / (1 + (
                            door.first_room_door_distance_values[incoming_delay_line.start.index - 30] /
                            door.first_room_door_distance_values[incoming_delay_line.start.index - 30]))
                    incoming_delay_line.buffer.appendleft(outgoing_sum * directivity_calculator(incoming_delay_line.start, door, self.directivity_mode))
            output_sum = 0.0
            for incoming_delay_line in door.incoming_delay_lines:
                if incoming_delay_line.start is source or (20 < incoming_delay_line.start.index):
                    output_sum += incoming_delay_line.buffer.pop() * directivity_calculator(incoming_delay_line.start, door, self.directivity_mode)
            door.input_output.append(output_sum)

            # For door as source in second room:
            # First energy factor here!
            for outgoing_delay_line in door.outgoing_delay_lines:
                if outgoing_delay_line.end is not microphone:
                    if outgoing_delay_line.end.index < 10:
                        outgoing_delay_line.buffer.appendleft(door.input_output[-1] * directivity_calculator(outgoing_delay_line.end, door, self.directivity_mode) *
                                                              0.5 * math.sqrt(door.area / self.total_area) *
                                                              1 / (door.second_room_distance_values[
                            outgoing_delay_line.end.index]))
                        for incoming_delay_line in outgoing_delay_line.end.incoming_delay_lines:
                            incoming_delay_line.buffer[-1] += outgoing_delay_line.buffer[-1]
                        outgoing_delay_line.buffer.pop()
            door.input_output.pop()
            # For scattering nodes in second room:
            for node in second_room.nodes:
                for i in range(0, len(node.outgoing_delay_lines)):
                    incoming_sum = 0.0
                    for j in range(0, len(node.incoming_delay_lines)):
                        incoming_sum += SCATTERING_MATRIX[i][j] * \
                                        node.incoming_delay_lines[j].buffer[-1]
                    if node.index % 10 == 1:
                        incoming_sum = incoming_sum * math.sqrt(1 - second_room.common_wall_absorption)
                    else:
                        incoming_sum = incoming_sum * second_room.WALL_REFLECTION_COEFFICIENT
                    node.outgoing_delay_lines[i].buffer.appendleft(incoming_sum)
                for delay_line in node.incoming_delay_lines:
                    delay_line.buffer.pop()

            # For door as microphone:
            for incoming_delay_line in door.incoming_delay_lines:
                if incoming_delay_line.start is not source and incoming_delay_line.start.index < 10:
                    outgoing_sum = 0.0
                    for outgoing_delay_line in incoming_delay_line.start.outgoing_delay_lines:
                        outgoing_sum += outgoing_delay_line.buffer[0]
                    outgoing_sum = outgoing_sum * (2 / 5) * second_room.WALL_REFLECTION_COEFFICIENT
                    incoming_delay_line.buffer.appendleft(outgoing_sum * directivity_calculator(incoming_delay_line.start, door, self.directivity_mode))
            output_sum = 0.0
            for incoming_delay_line in door.incoming_delay_lines:
                if incoming_delay_line.start is not source and incoming_delay_line.start.index < 10:
                    output_sum += incoming_delay_line.buffer.pop() * directivity_calculator(incoming_delay_line.start, door, self.directivity_mode)
            door.door_last_output.append(output_sum)

            # For door as source in first room:
            # Second energy factor here!
            for outgoing_delay_line in door.outgoing_delay_lines:
                if outgoing_delay_line.end is not microphone and 10 < outgoing_delay_line.end.index < 20:
                    outgoing_delay_line.buffer.appendleft(door.door_last_output[-1] * directivity_calculator(outgoing_delay_line.end, door, self.directivity_mode) *
                                                          0.5 *
                                                          1 / (
                                                              door.first_room_distance_values[
                                                                  outgoing_delay_line.end.index - 10]))
                    for incoming_delay_line in outgoing_delay_line.end.incoming_delay_lines:
                        incoming_delay_line.buffer[-1] += outgoing_delay_line.buffer[-1]
                    outgoing_delay_line.buffer.pop()
                elif outgoing_delay_line.end is not microphone and outgoing_delay_line.end.index > 30:

                    outgoing_delay_line.buffer.appendleft(door.door_last_output[-1] * directivity_calculator(outgoing_delay_line.end, door, self.directivity_mode) *
                                                          0.5 *
                                                          1 / (
                                                              door.first_room_door_distance_values[
                                                                  outgoing_delay_line.end.index - 30]))
                    for incoming_delay_line in outgoing_delay_line.end.incoming_delay_lines:
                        incoming_delay_line.buffer[-1] += outgoing_delay_line.buffer[-1]
                    outgoing_delay_line.buffer.pop()
                elif outgoing_delay_line.end is microphone:
                    outgoing_delay_line.buffer.appendleft(
                        door.door_last_output[-1] * 1 / (door.first_room_distance_values[0]) * directivity_calculator(outgoing_delay_line.end, door, self.directivity_mode))
            door.door_last_output.pop()

            # For scattering nodes in first room door feedback
            for node in self.doordoor_nodes:
                for i in range(0, len(node.outgoing_delay_lines)):
                    incoming_sum = 0.0
                    for j in range(0, len(node.incoming_delay_lines)):
                        incoming_sum += SCATTERING_MATRIX[i][j] * \
                                        node.incoming_delay_lines[j].buffer[-1]
                    if node.index % 10 == 3:
                        incoming_sum = incoming_sum * math.sqrt(1 - self.common_wall_absorption)
                    else:
                        incoming_sum = incoming_sum * self.WALL_REFLECTION_COEFFICIENT
                    node.outgoing_delay_lines[i].buffer.appendleft(incoming_sum)
                for delay_line in node.incoming_delay_lines:
                    delay_line.buffer.pop()

            # For scattering nodes in first room (door as source):
            for node in self.doormic_nodes:
                for i in range(0, len(node.outgoing_delay_lines)):
                    incoming_sum = 0.0
                    for j in range(0, len(node.incoming_delay_lines)):
                        incoming_sum += SCATTERING_MATRIX[i][j] * \
                                        node.incoming_delay_lines[j].buffer[-1]
                    if node.index % 10 == 3:
                        incoming_sum = incoming_sum * math.sqrt(1 - self.common_wall_absorption)
                    else:
                        incoming_sum = incoming_sum * self.WALL_REFLECTION_COEFFICIENT
                    node.outgoing_delay_lines[i].buffer.appendleft(incoming_sum)
                for delay_line in node.incoming_delay_lines:
                    delay_line.buffer.pop()

            # For microphone:
            for incoming_delay_line in microphone.incoming_delay_lines:
                if incoming_delay_line.start is not source and incoming_delay_line.start is not door:
                    if incoming_delay_line.start.index > 10:
                        outgoing_sum = 0.0
                        for outgoing_delay_line in incoming_delay_line.start.outgoing_delay_lines:
                            outgoing_sum += outgoing_delay_line.buffer[0]
                        if incoming_delay_line.start.index % 10 == 3:
                            outgoing_sum = outgoing_sum * \
                                           (2 / 5) * math.sqrt(1 - self.common_wall_absorption) * \
                                           1 / (microphone.distance_values[incoming_delay_line.start.index - 10])
                        else:
                            outgoing_sum = outgoing_sum * \
                                           (2 / 5) * self.WALL_REFLECTION_COEFFICIENT * \
                                           1 / (microphone.distance_values[incoming_delay_line.start.index - 10])
                        incoming_delay_line.buffer.appendleft(outgoing_sum)
                    else:
                        outgoing_sum = 0.0
                        for outgoing_delay_line in incoming_delay_line.start.outgoing_delay_lines:
                            outgoing_sum += outgoing_delay_line.buffer[0]
                        if incoming_delay_line.start.index % 10 == 3:
                            outgoing_sum = outgoing_sum * \
                                           (2 / 5) * math.sqrt(1 - self.common_wall_absorption) * \
                                           1 / (1 + (microphone.distance_values[incoming_delay_line.start.index] /
                                                     source.distance_values[incoming_delay_line.start.index]))
                        else:
                            outgoing_sum = outgoing_sum * \
                                           (2 / 5) * self.WALL_REFLECTION_COEFFICIENT * \
                                           1 / (1 + (microphone.distance_values[incoming_delay_line.start.index] /
                                                     source.distance_values[incoming_delay_line.start.index]))
                        incoming_delay_line.buffer.appendleft(outgoing_sum)
            output_sum = 0.0
            for incoming_delay_line in microphone.incoming_delay_lines:
                output_sum += incoming_delay_line.buffer.pop()
            microphone.output.append(output_sum)
        elif self.is_source_in_room and not self.is_microphone_in_room:
            for outgoing_delay_line in source.outgoing_delay_lines:
                if outgoing_delay_line.end is not door:
                    outgoing_delay_line.buffer.appendleft(
                        source.input[-1] * 0.5 * 1 / (source.distance_values[outgoing_delay_line.end.index]))
                    for incoming_delay_line in outgoing_delay_line.end.incoming_delay_lines:
                        incoming_delay_line.buffer[-1] += outgoing_delay_line.buffer[-1]
                    outgoing_delay_line.buffer.pop()
                else:
                    outgoing_delay_line.buffer.appendleft(
                        math.sqrt(door.area / self.total_area) * source.input[
                            -1] * 1 / (source.distance_values[0]))
            source.input.appendleft(0.0)
            source.input.pop()
            # For scattering nodes:
            for node in self.nodes:
                for i in range(0, len(node.outgoing_delay_lines)):
                    incoming_sum = 0.0
                    for j in range(0, len(node.incoming_delay_lines)):
                        incoming_sum += SCATTERING_MATRIX[i][j] * \
                                        node.incoming_delay_lines[j].buffer[-1]
                        if node.index % 10 == 3:
                            incoming_sum = incoming_sum * math.sqrt(1 - self.common_wall_absorption)
                    else:
                        incoming_sum = incoming_sum * self.WALL_REFLECTION_COEFFICIENT
                    node.outgoing_delay_lines[i].buffer.appendleft(incoming_sum)
                for delay_line in node.incoming_delay_lines:
                    delay_line.buffer.pop()
            # For door:
            for incoming_delay_line in door.incoming_delay_lines:
                if incoming_delay_line.start is not source and incoming_delay_line.start.index != 3:
                    outgoing_sum = 0.0
                    for outgoing_delay_line in incoming_delay_line.start.outgoing_delay_lines:
                        outgoing_sum += outgoing_delay_line.buffer[0]

                    outgoing_sum = outgoing_sum * \
                                   (2 / 5) * self.WALL_REFLECTION_COEFFICIENT * \
                                   1 / (1 + (door.first_room_distance_values[incoming_delay_line.start.index] /
                                             source.distance_values[incoming_delay_line.start.index]))
                    incoming_delay_line.buffer.appendleft(outgoing_sum)
            output_sum = 0.0
            for incoming_delay_line in door.incoming_delay_lines:
                if incoming_delay_line.start.index != 3:
                    output_sum += incoming_delay_line.buffer.pop()
            door.input_output.append(output_sum)
        elif not self.is_source_in_room and self.is_microphone_in_room:
            # For source:
            for outgoing_delay_line in door.outgoing_delay_lines:
                if outgoing_delay_line.end is not microphone and outgoing_delay_line.end.index != 3:
                    outgoing_delay_line.buffer.appendleft(door.input_output[-1] *
                                                          0.5 *
                                                          1 / (door.first_room_distance_values[
                        outgoing_delay_line.end.index]))
                    for incoming_delay_line in outgoing_delay_line.end.incoming_delay_lines:
                        incoming_delay_line.buffer[-1] += outgoing_delay_line.buffer[-1]
                    outgoing_delay_line.buffer.pop()
                else:
                    outgoing_delay_line.buffer.appendleft(door.input_output[-1] *
                                                          1 / (door.first_room_distance_values[0]))
            door.input_output.appendleft(0.0)
            door.input_output.pop()
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
                if incoming_delay_line.start is not door:
                    outgoing_sum = 0.0
                    for outgoing_delay_line in incoming_delay_line.start.outgoing_delay_lines:
                        outgoing_sum += outgoing_delay_line.buffer[0]
                    outgoing_sum = outgoing_sum * \
                                   (2 / 5) * self.WALL_REFLECTION_COEFFICIENT * \
                                   1 / (1 + (microphone.distance_values[incoming_delay_line.start.index] /
                                             door.first_room_distance_values[incoming_delay_line.start.index]))
                    incoming_delay_line.buffer.appendleft(outgoing_sum)
            output_sum = 0.0
            for incoming_delay_line in microphone.incoming_delay_lines:
                output_sum += incoming_delay_line.buffer.pop()
            microphone.output.appendleft(output_sum)


class SecondRoom(object):
    def __init__(self, x, y, z
                 , absorption_coefficient
                 , microphone
                 , source
                 , door
                 , room):

        """
        Creates the second room with the given parameters:
        1- Absorption Coefficient
        2- Microphone
        3- Source
        4- Door
        5- First Room

        First, the wall reflection coefficient is calculated from the absorption coefficient.
        Secondly, Room coordinates are created with numpy arrays.
        Thirdly, image list, and node list are initialized as None. First Room is initialized.
        Finally, positions of the microphone, source, and the door are checked.
        If placed correctly, the room object is created.

        If microphone and/or source does not reside in the room, insert None to create the room.

        """
        self.WALL_REFLECTION_COEFFICIENT = -math.sqrt(1 - absorption_coefficient)
        self.absorption_coefficient = absorption_coefficient

        self.x = np.array([room.x[1], room.x[1] + x])
        self.y = np.array([0., y])
        self.z = np.array([0., z])

        self.images = []
        self.nodes = []

        self.first_room = room

        if microphone is not None:
            if microphone.x > self.x[1] or microphone.y > y or microphone.z > z \
                    or microphone.x < self.x[0] or microphone.y < 0 or microphone.z < 0:
                raise AssertionError("Microphone is out of bounds!")

        if source is not None:
            if source.x > self.x[1] or source.y > y or source.z > z \
                    or source.x < self.x[0] or source.y < 0 or source.z < 0:
                raise AssertionError("Source is out of bounds!")

        if door.x != self.x[0]:
            raise AssertionError("Door is not placed correctly!")

        self.microphone = microphone
        self.source = source
        self.door = door

        self.is_microphone_in_room = microphone is not None
        self.is_source_in_room = source is not None

        self.total_area = self.y[1] * self.z[1]
        self.common_wall_absorption = (self.door.area + (
                self.total_area - self.door.area) * self.absorption_coefficient) / self.total_area

        print(self.total_area)
        self.max_area = 2 * self.y[1] * self.z[1] + 2 * self.x[1] * self.z[1] + 2 * self.y[1] * self.x[1]

        print(self.max_area)

    def find_images(self):
        """
        Image method finds the images in the room with respect to source and microphone existence.

        If the source is in the room, source's images are found no matter what.
        If the microphone is in the room, door's images are found.
        If both source and microphone are missing, there is no need to find images as scattering nodes of an empty room
        will be placed at the center of walls.
        """

        source = self.source

        if self.is_source_in_room:

            self.images.append(np.array([2 * self.x[0] - source.x, source.y, source.z]))
            self.images.append(np.array([2 * self.x[1] - source.x, source.y, source.z]))

            self.images.append(np.array([source.x, -source.y, source.z]))
            self.images.append(np.array([source.x, 2 * self.y[1] - source.y, source.z]))

            self.images.append(np.array([source.x, source.y, -source.z]))
            self.images.append(np.array([source.x, source.y, 2 * self.z[1] - source.z]))

        elif not self.is_source_in_room and self.is_microphone_in_room:

            self.images.append(np.array([2 * self.x[1] - self.door.x, self.door.y, self.door.z]))

            self.images.append(np.array([self.door.x, -self.door.y, self.door.z]))
            self.images.append(np.array([self.door.x, 2 * self.y[1] - self.door.y, self.door.z]))

            self.images.append(np.array([self.door.x, self.door.y, - self.door.z]))
            self.images.append(np.array([self.door.x, self.door.y, 2 * self.z[1] - self.door.z]))

    def find_sdn_nodes(self):
        microphone = self.microphone
        # y_offset = 5
        # z_offset = (13 / 2)
        y_offset = 0
        z_offset = 0

        if self.is_source_in_room and self.is_microphone_in_room:
            self.sdn_finder(microphone)
        elif not self.is_source_in_room and not self.is_microphone_in_room:

            new_node = ScatteringNode(
                np.array([self.x[0], (self.y[1] - self.y[0]) / 2 + y_offset, (self.z[1] - self.z[0]) / 2 + z_offset]),
                1)
            self.nodes.append(new_node)

            new_node = ScatteringNode(
                np.array([(self.x[1] - self.x[0]) / 2 + self.x[0], self.y[1] + y_offset,
                          (self.z[1] - self.z[0]) / 2 + z_offset]),
                2)
            self.nodes.append(new_node)

            new_node = ScatteringNode(
                np.array([self.x[1], (self.y[1] - self.y[0]) / 2 + y_offset, (self.z[1] - self.z[0]) / 2 + z_offset]),
                3)
            self.nodes.append(new_node)

            new_node = ScatteringNode(
                np.array([(self.x[1] - self.x[0]) / 2 + self.x[0], self.y[0] + y_offset,
                          (self.z[1] - self.z[0]) / 2 + z_offset]),
                4)
            self.nodes.append(new_node)

            new_node = ScatteringNode(
                np.array([(self.x[1] - self.x[0]) / 2 + self.x[0], (self.y[1] - self.y[0]) / 2 + y_offset,
                          self.z[0] + z_offset]),
                5)
            self.nodes.append(new_node)

            new_node = ScatteringNode(
                np.array([(self.x[1] - self.x[0]) / 2 + self.x[0], (self.y[1] - self.y[0]) / 2 + y_offset,
                          self.z[1] + z_offset]),
                6)
            self.nodes.append(new_node)

        elif not self.is_source_in_room and self.is_microphone_in_room:
            new_node = ScatteringNode(np.array([self.x[0], (self.y[1] - self.y[0]) / 2, (self.z[1] - self.z[0]) / 2]),
                                      1)
            self.nodes.append(new_node)
            self.sdn_finder(microphone)

        else:
            new_node = ScatteringNode(np.array([self.x[0], (self.y[1] - self.y[0]) / 2, (self.z[1] - self.z[0]) / 2]),
                                      1)
            self.nodes.append(new_node)
            self.sdn_finder(self.door)

    def sdn_finder(self, target_object):
        x = self.x[1] - self.x[0]
        y = self.y[1]
        z = self.z[1]

        for image in self.images:
            # to-do: Find a simpler way of performing SDN_Calculation
            if image[0] < self.x[0]:
                # Plane points and normal vector
                point1 = np.array([self.x[0], random.uniform(0, y), random.uniform(0, z)])
                point2 = np.array([self.x[0], random.uniform(0, y), random.uniform(0, z)])
                point3 = np.array([self.x[0], random.uniform(0, y), random.uniform(0, z)])

                intersection_point = line_intersection(image, target_object, point1, point2, point3)

                new_node = ScatteringNode(intersection_point, 1)

                is_already_available = False
                for node in self.nodes:
                    if node.index == 1:
                        is_already_available = True
                if not is_already_available:
                    self.nodes.append(new_node)

            elif image[1] >= y:
                # Plane points and normal vector
                point1 = np.array([random.uniform(0, x), y, random.uniform(0, z)])
                point2 = np.array([random.uniform(0, x), y, random.uniform(0, z)])
                point3 = np.array([random.uniform(0, x), y, random.uniform(0, z)])

                intersection_point = line_intersection(image, target_object, point1, point2, point3)

                new_node = ScatteringNode(intersection_point, 2)
                self.nodes.append(new_node)

            elif image[0] > self.x[1]:
                # Plane points and normal vector
                point1 = np.array([self.x[1], random.uniform(0, y), random.uniform(0, z)])
                point2 = np.array([self.x[1], random.uniform(0, y), random.uniform(0, z)])
                point3 = np.array([self.x[1], random.uniform(0, y), random.uniform(0, z)])

                intersection_point = line_intersection(image, target_object, point1, point2, point3)

                new_node = ScatteringNode(intersection_point, 3)
                self.nodes.append(new_node)

            elif image[1] <= 0:
                # Plane points and normal vector
                point1 = np.array([random.uniform(0, x), 0, random.uniform(0, z)])
                point2 = np.array([random.uniform(0, x), 0, random.uniform(0, z)])
                point3 = np.array([random.uniform(0, x), 0, random.uniform(0, z)])

                intersection_point = line_intersection(image, target_object, point1, point2, point3)

                new_node = ScatteringNode(intersection_point, 4)
                self.nodes.append(new_node)

            elif image[2] < 0:
                # Plane points and normal vector
                point1 = np.array([random.uniform(0, x), random.uniform(0, y), 0])
                point2 = np.array([random.uniform(0, x), random.uniform(0, y), 0])
                point3 = np.array([random.uniform(0, x), random.uniform(0, y), 0])

                intersection_point = line_intersection(image, target_object, point1, point2, point3)

                new_node = ScatteringNode(intersection_point, 5)
                self.nodes.append(new_node)

            elif image[2] > z:
                # Plane points and normal vector
                point1 = np.array([random.uniform(0, x), random.uniform(0, y), z])
                point2 = np.array([random.uniform(0, x), random.uniform(0, y), z])
                point3 = np.array([random.uniform(0, x), random.uniform(0, y), z])

                intersection_point = line_intersection(image, target_object, point1, point2, point3)

                new_node = ScatteringNode(intersection_point, 6)
                self.nodes.append(new_node)

            else:
                pass

            # Order SDN nodes by their indexes
        new_nodes = []
        for node in self.nodes:
            new_nodes.insert(node.index - 1, node)
        self.nodes = new_nodes

    def create_delay_lines(self):

        source = self.source
        microphone = self.microphone
        door = self.door

        node_list = self.nodes

        delay_lines = []

        if not self.is_source_in_room and not self.is_microphone_in_room:
            door = self.door

            node_list = self.nodes

            delay_lines = []

            # Outgoing_delay_line from door to nodes
            for node in node_list:
                if node.index == 1:
                    continue
                delay_line = DelayLine(door, node)
                delay_lines.append(delay_line)
                door.outgoing_delay_lines.append(delay_line)

            # Creation of outgoing_delay_lines between nodes
            for i in range(0, len(node_list)):
                delay_line = DelayLine(node_list[i], door)
                delay_lines.append(delay_line)
                if i != 0:
                    door.incoming_delay_lines.append(delay_line)
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
                    if delay_line.start is door:
                        continue
                    if delay_line.end is node:
                        node.incoming_delay_lines.append(delay_line)
        elif not self.is_source_in_room and self.is_microphone_in_room:
            delay_line = DelayLine(door, microphone)
            delay_lines.append(delay_line)
            door.outgoing_delay_lines.append(delay_line)

            for node in node_list:
                delay_line = DelayLine(door, node)
                delay_lines.append(delay_line)
                door.outgoing_delay_lines.append(delay_line)

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
            for node in node_list:
                for delay_line in delay_lines:
                    if delay_line.start is door:
                        continue
                    if delay_line.end is node:
                        node.incoming_delay_lines.append(delay_line)
            for delay_line in delay_lines:
                if delay_line.end is microphone:
                    microphone.incoming_delay_lines.append(delay_line)
        elif self.is_source_in_room and not self.is_microphone_in_room:

            delay_line = DelayLine(source, door)
            delay_lines.append(delay_line)
            source.outgoing_delay_lines.append(delay_line)

            for node in node_list:
                delay_line = DelayLine(source, node)
                delay_lines.append(delay_line)
                source.outgoing_delay_lines.append(delay_line)

            for i in range(0, len(node_list)):
                delay_line = DelayLine(node_list[i], door)
                delay_lines.append(delay_line)
                for j in range(0, len(node_list)):
                    if i == j:
                        continue
                    else:
                        delay_line = DelayLine(node_list[i], node_list[j])
                        delay_lines.append(delay_line)
                        node_list[i].outgoing_delay_lines.append(delay_line)
            for node in node_list:
                for delay_line in delay_lines:
                    if delay_line.start is source:
                        continue
                    if delay_line.end is node:
                        node.incoming_delay_lines.append(delay_line)
            for delay_line in delay_lines:
                if delay_line.end is door:
                    door.incoming_delay_lines.append(delay_line)
        else:

            # Outgoing_delay_line from source to microphone
            delay_line = DelayLine(source, microphone)
            delay_lines.append(delay_line)
            source.outgoing_delay_lines.append(delay_line)

            # Outgoing_delay_line from source to door
            delay_line = DelayLine(source, door)
            delay_lines.append(delay_line)
            source.outgoing_delay_lines.append(delay_line)
            door.incoming_delay_lines.append(delay_line)

            # Outgoing_delay_lines from source to microphone
            for node in node_list:
                delay_line = DelayLine(source, node)
                delay_lines.append(delay_line)
                source.outgoing_delay_lines.append(delay_line)

            # Outgoing delay_line from door to microphone
            delay_line = DelayLine(door, microphone)
            delay_lines.append(delay_line)
            door.outgoing_delay_lines.append(delay_line)

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
        door = self.door
        microphone = self.microphone

        if not self.is_source_in_room and not self.is_microphone_in_room:

            for node in self.nodes:
                door.second_room_distance_values[node.index] = np.linalg.norm(door.position - node.position)

        elif not self.is_source_in_room and self.is_microphone_in_room:

            for node in self.nodes:
                door.second_room_distance_values[node.index] = np.linalg.norm(door.position - node.position)
                microphone.distance_values[node.index] = np.linalg.norm(microphone.position - node.position)
            door.second_room_distance_values[0] = np.linalg.norm(door.position - microphone.position)
            microphone.distance_values[0] = np.linalg.norm(microphone.position - door.position)

        elif self.is_source_in_room and not self.is_microphone_in_room:

            for node in self.nodes:
                door.second_room_distance_values[node.index] = np.linalg.norm(door.position - node.position)
                source.distance_values[node.index] = np.linalg.norm(source.position - node.position)
            door.second_room_distance_values[0] = np.linalg.norm(door.position - source.position)
            source.distance_values[0] = np.linalg.norm(door.position - source.position)

        else:

            for node in self.nodes:
                source.distance_values[node.index] = np.linalg.norm(source.position - node.position)
                microphone.distance_values[node.index] = np.linalg.norm(microphone.position - node.position)

            source.distance_values[0] = np.linalg.norm(source.position - microphone.position)
            microphone.distance_values[0] = np.linalg.norm(microphone.position - source.position)

            source.distance_values.append(np.linalg.norm(source.position - door.position))
            microphone.distance_values.append(np.linalg.norm(microphone.position - door.position))

            door.second_room_distance_values[0] = np.linalg.norm(door.position - microphone.position)

    def tick_function(self):

        source = self.source
        door = self.door
        microphone = self.microphone
        first_room = self.first_room

        if not self.is_source_in_room and not self.is_microphone_in_room:
            pass

        elif not self.is_source_in_room and self.is_microphone_in_room:
            # This is the only working solution for the different volume case
            # For door:
            for outgoing_delay_line in door.outgoing_delay_lines:
                if outgoing_delay_line.end is not microphone and outgoing_delay_line.end.index != 1:
                    outgoing_delay_line.buffer.appendleft(
                        math.sqrt(door.area / self.total_area) * door.input_output[-1] *
                        0.5 *
                        1 / (door.second_room_distance_values[
                            outgoing_delay_line.end.index]))
                    for incoming_delay_line in outgoing_delay_line.end.incoming_delay_lines:
                        incoming_delay_line.buffer[-1] += outgoing_delay_line.buffer[-1]
                    outgoing_delay_line.buffer.pop()
                elif outgoing_delay_line.end is microphone:
                    outgoing_delay_line.buffer.appendleft(
                        math.sqrt(door.area / self.total_area) * door.input_output[-1] *
                        1 / (door.second_room_distance_values[0]))
            door.input_output.pop()
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
                if incoming_delay_line.start is not door:
                    outgoing_sum = 0.0
                    for outgoing_delay_line in incoming_delay_line.start.outgoing_delay_lines:
                        outgoing_sum += outgoing_delay_line.buffer[0]
                    outgoing_sum = outgoing_sum * \
                                   (2 / 5) * self.WALL_REFLECTION_COEFFICIENT * \
                                   1 / (1 + (microphone.distance_values[incoming_delay_line.start.index] /
                                             door.first_room_distance_values[incoming_delay_line.start.index]))
                    incoming_delay_line.buffer.appendleft(outgoing_sum)
            output_sum = 0.0
            for incoming_delay_line in microphone.incoming_delay_lines:
                output_sum += incoming_delay_line.buffer.pop()
            microphone.output.append(output_sum)
        elif self.is_source_in_room and not self.is_microphone_in_room:
            # This is outdated and not working please skip this!
            # For source:
            for outgoing_delay_line in source.outgoing_delay_lines:
                if outgoing_delay_line.end is not door:
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
            for incoming_delay_line in door.incoming_delay_lines:
                if incoming_delay_line.start is not source:
                    outgoing_sum = 0.0
                    for outgoing_delay_line in incoming_delay_line.start.outgoing_delay_lines:
                        outgoing_sum += outgoing_delay_line.buffer[0]
                    outgoing_sum = outgoing_sum * \
                                   (2 / 5) * self.WALL_REFLECTION_COEFFICIENT * \
                                   1 / (1 + (door.second_room_distance_values[incoming_delay_line.start.index] /
                                             source.distance_values[incoming_delay_line.start.index]))
                    incoming_delay_line.buffer.appendleft(outgoing_sum)
            output_sum = 0.0
            for incoming_delay_line in door.incoming_delay_lines:
                if incoming_delay_line.start is not source:
                    if incoming_delay_line.start.index == 1:
                        continue
                    output_sum += incoming_delay_line.buffer.pop()
            door.input_output.append(output_sum)
        else:
            # For source:
            for outgoing_delay_line in source.outgoing_delay_lines:
                if outgoing_delay_line.end is not microphone and outgoing_delay_line.end is not door:
                    outgoing_delay_line.buffer.appendleft(source.input[-1] *
                                                          0.5 *
                                                          1 / (source.distance_values[outgoing_delay_line.end.index]))
                    for incoming_delay_line in outgoing_delay_line.end.incoming_delay_lines:
                        incoming_delay_line.buffer[-1] += outgoing_delay_line.buffer[-1]
                    outgoing_delay_line.buffer.pop()
                elif outgoing_delay_line.end is microphone:
                    outgoing_delay_line.buffer.appendleft(source.input[-1] *
                                                          1 / (source.distance_values[0]))
                elif outgoing_delay_line.end is door:
                    outgoing_delay_line.buffer.appendleft(source.input[-1] *
                                                          1 / (source.distance_values[7]))
            source.input.appendleft(0.0)
            source.input.pop()
            # For scattering nodes in first room:
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
            # For door as source:
            for incoming_delay_line in door.incoming_delay_lines:
                if incoming_delay_line.start is source:
                    door.input_output.appendleft(incoming_delay_line.buffer.pop())
            for outgoing_delay_line in door.outgoing_delay_lines:
                if outgoing_delay_line.end is not microphone:
                    outgoing_delay_line.buffer.appendleft(door.input_output[-1] *
                                                          0.5 *
                                                          1 / (door.first_room_distance_values[
                        outgoing_delay_line.end.index]))
                    for incoming_delay_line in outgoing_delay_line.end.incoming_delay_lines:
                        incoming_delay_line.buffer[-1] += outgoing_delay_line.buffer[-1]
                    outgoing_delay_line.buffer.pop()
            door.input_output.pop()
            # For scattering nodes in first room:
            for node in first_room.nodes:
                for i in range(0, len(node.outgoing_delay_lines)):
                    incoming_sum = 0.0
                    for j in range(0, len(node.incoming_delay_lines)):
                        incoming_sum += SCATTERING_MATRIX[i][j] * \
                                        node.incoming_delay_lines[j].buffer[-1]
                    incoming_sum = incoming_sum * first_room.WALL_REFLECTION_COEFFICIENT
                    node.outgoing_delay_lines[i].buffer.appendleft(incoming_sum)
                for delay_line in node.incoming_delay_lines:
                    delay_line.buffer.pop()
            # For door as microphone:
            for incoming_delay_line in door.incoming_delay_lines:
                if incoming_delay_line.start is not source and incoming_delay_line.start.index != 3:
                    outgoing_sum = 0.0
                    for outgoing_delay_line in incoming_delay_line.start.outgoing_delay_lines:
                        outgoing_sum += outgoing_delay_line.buffer[0]
                    outgoing_sum = outgoing_sum * \
                                   (2 / 5) * first_room.WALL_REFLECTION_COEFFICIENT * \
                                   1 / (1 + (door.first_room_distance_values[incoming_delay_line.start.index] /
                                             door.first_room_distance_values[incoming_delay_line.start.index]))
                    incoming_delay_line.buffer.appendleft(outgoing_sum)
            # For door as source:
            output_sum_for_first_room = 0.0
            for incoming_delay_line in door.incoming_delay_lines:
                if incoming_delay_line.start is not source and incoming_delay_line.start.index != 3:
                    output_sum_for_first_room += incoming_delay_line.buffer.pop()
            for outgoing_delay_line in door.outgoing_delay_lines:
                if outgoing_delay_line.end is microphone:
                    outgoing_delay_line.buffer.append(
                        output_sum_for_first_room * 1 / door.second_room_distance_values[0])
            # For microphone:
            for incoming_delay_line in microphone.incoming_delay_lines:
                if incoming_delay_line.start is not source and incoming_delay_line.start is not door:
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


class Door(object):
    def __init__(self, position, area):
        self.position = position

        self.x = position[0]
        self.y = position[1]
        self.z = position[2]

        self.incoming_delay_lines = []
        self.outgoing_delay_lines = []

        self.input_output = collections.deque([])

        self.door_last_output = collections.deque([])

        self.first_room_distance_values = [None,
                                           None,
                                           None,
                                           None,
                                           None,
                                           None,
                                           None]

        self.second_room_distance_values = [None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None,
                                            None]

        self.first_room_source_distance_values = [None,
                                                  None,
                                                  None,
                                                  None,
                                                  None,
                                                  None,
                                                  None]

        self.first_room_door_distance_values = [None,
                                                None,
                                                None,
                                                None,
                                                None,
                                                None,
                                                None]

        self.area = area

        self.index = None

    def __repr__(self):
        return "Door"


class Microphone(object):
    def __init__(self, position):
        self.position = position

        self.x = position[0]
        self.y = position[1]
        self.z = position[2]

        self.output = collections.deque([])

        self.incoming_delay_lines = []

        self.distance_values = [None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None]

        self.door_distance_values = [None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None]

        self.index = None

    def __repr__(self):
        return "Microphone"


class Source(object):
    def __init__(self, position):
        self.position = position

        self.x = position[0]
        self.y = position[1]
        self.z = position[2]

        self.outgoing_delay_lines = []

        self.input = collections.deque([])

        self.distance_values = [None,
                                None,
                                None,
                                None,
                                None,
                                None,
                                None]

        self.door_distance_values = [None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None]
        self.index = None

    def __repr__(self):
        return "Source"


class ScatteringNode(object):
    def __init__(self, position, index=None):
        self.position = position

        self.x = position[0]
        self.y = position[1]
        self.z = position[2]

        self.index = index

        self.incoming_delay_lines = []

        self.outgoing_delay_lines = []

    def __repr__(self):
        return "SN Index: " + str(self.index)


class DelayLine(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

        self.distance = np.linalg.norm(start.position - end.position)

        length = int((SAMPLING_RATE * self.distance) / SPEED_OF_SOUND)

        self.length = length

        self.buffer = collections.deque(np.zeros(self.length, dtype=np.float))

    def __repr__(self):
        return "Delay Line from " + str(self.start) + " to " + str(self.end)


class SoundFileRW(object):
    def __init__(self):
        self.data = None
        self.rate = None

    def read_sound_file(self, filename, source, operation=0):
        # If impulse is given:

        # self.rate, self.data = wavfile.read(filename)
        self.rate = 44100
        self.data = sig.unit_impulse(1)
        self.data.flags.writeable = False
        self.data = self.data.astype(float)

        if operation == 1:
            # self.rate, self.data = wavfile.read(filename)
            self.data = sig.unit_impulse(1)

        source.input = collections.deque(np.flip(self.data))

    def write_sound_file(self, filename, microphone):
        self.data = np.array(microphone.output)
        self.data = self.data / np.max(np.abs(self.data))
        self.data = np.trim_zeros(self.data, 'f')

        # If you want to see impulse:
        # plt.plot(self.data)
        # plt.show()

        wavfile.write(filename, self.rate, self.data)
        # self.data = np.asarray(microphone.output, dtype=np.int16)
        return self.rate, self.data

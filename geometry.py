from __future__ import annotations  # This allows forward references in type hints
import numpy as np
from typing import Dict, List


class Room:
    """
    Class defining a room with some propoerties that can be controlled
    """

    def __init__(self, lx, ly, lz):
        self.label = "cuboid"
        self.nWalls = 6
        self.x = lx
        self.y = ly
        self.z = lz
        self.wallAttenuation = []  # this is a list
        self.wallFilters = dict()  # this is a dictionary
        self.walls = {}
        self.walls['south'] = Wall(Point(0, 0, 0), Point(self.x, 0, 0), Point(0, 0, self.z))
        self.walls['north'] = Wall(Point(0, self.y, 0), Point(0, self.y, self.z), Point(self.x, self.y, 0))
        self.walls['west'] = Wall(Point(0, 0, 0), Point(0, self.y, 0), Point(0, 0, self.z))
        self.walls['east'] = Wall(Point(self.x, 0, 0), Point(self.x, self.y, 0), Point(self.x, self.y, self.z))
        self.walls['ceiling'] = Wall(Point(0, 0, self.z), Point(0, self.y, self.z), Point(self.x, self.y, self.z))
        self.walls['floor'] = Wall(Point(0, 0, 0), Point(0, self.y, 0), Point(self.x, self.y, 0))

    def set_microphone(self, mx, my, mz):
        self.mx = mx
        self.my = my
        self.mz = mz
        self.micPos = Point(mx, my, mz)

    def set_source(self, sx, sy, sz, signal, Fs=44100):
        self.source = Source(sx, sy, sz, signal, Fs)
        # Calculate image sources for all walls immediately
        img_sources = ImageSource(self.walls, self.source.srcPos, self.micPos)
        img_sources.findImageSources()


class Source:
    def __init__(self, sx, sy, sz, signal, Fs):
        self.sx = sx
        self.sy = sy
        self.sz = sz
        self.srcPos = Point(sx, sy, sz)
        self.signal = signal
        self.Fs = Fs
        # self.directivity = 1.0
        # self.heading = 0.0

class ImageSource:
    def __init__(self, walls: Dict[str, Wall], srcPos: Point, micPos: Point):
        self.walls = walls
        self.srcPos = srcPos
        self.micPos = micPos
        self.imageSourcePos = None  # Will be set by IS_1st_order

    def findImageSources(self):
        for wall_label, wall in self.walls.items():
            self.imageSourcePos = self.IS_1st_order(wall)
            wall.IMS = self.imageSourcePos
            wall.node_positions = self.findLineIntersection()
            print(f"Set node position for {wall_label}: {wall.node_positions.__dict__}")  # Debug print

    def IS_1st_order(self, wall):
        # find image source locations along the plane
        self.d = wall.plane_coeffs.d
        self.a = wall.plane_coeffs.a
        self.b = wall.plane_coeffs.b
        self.c = wall.plane_coeffs.c

        # Compute the distance from the point to the plane
        # Distance formula: (ax + by + cz + d) / sqrt(a^2 + b^2 + c^2)
        norm = self.a ** 2 + self.b ** 2 + self.c ** 2
        dist_to_plane = (self.a * self.srcPos.x + self.b * self.srcPos.y + self.c * self.srcPos.z + self.d) / norm

        # Compute the reflection point
        self.ImageSource_Pos = Point(0.0, 0.0, 0.0)
        self.ImageSource_Pos.x = self.srcPos.x - 2 * dist_to_plane * self.a
        self.ImageSource_Pos.y = self.srcPos.y - 2 * dist_to_plane * self.b
        self.ImageSource_Pos.z = self.srcPos.z - 2 * dist_to_plane * self.c

        return self.ImageSource_Pos

    def findLineIntersection(self):
        # two points are enough to define a line
        # equation of a line is (x-x1)/l = (y-y1)/m = (z-z1)/n = k
        posA = self.ImageSource_Pos
        posB = self.micPos
        l = posB.x - posA.x
        m = posB.y - posA.y
        n = posB.z - posA.z

        # replace x with kl + x1 etc and plug into ax + by + cz + d = 0 to find k
        k = -(self.a * posA.x + self.b * posA.y + self.c * posA.z +
              self.d) / (self.a * l + self.b * m + self.c * n)

        # plug in value of k into x = kl+x1 etc to find point of intersection
        interPos = Point(0.0, 0.0, 0.0)
        interPos.x = k * l + posA.x
        interPos.y = k * m + posA.y
        interPos.z = k * n + posA.z

        return interPos

    def findIntersectionPoint(self, point1: Point, point2: Point, wall: 'Wall') -> Point:
        """New method for ISM - takes explicit points"""
        # two points are enough to define a line
        # equation of a line is (x-x1)/l = (y-y1)/m = (z-z1)/n = k
        l = point2.x - point1.x
        m = point2.y - point1.y
        n = point2.z - point1.z

        # replace x with kl + x1 etc and plug into ax + by + cz + d = 0 to find k
        k = -(wall.plane_coeffs.a * point1.x + 
              wall.plane_coeffs.b * point1.y + 
              wall.plane_coeffs.c * point1.z +
              wall.plane_coeffs.d) / (wall.plane_coeffs.a * l + 
                                    wall.plane_coeffs.b * m + 
                                    wall.plane_coeffs.c * n)

        # plug in value of k into x = kl+x1 etc to find point of intersection
        interPos = Point(0.0, 0.0, 0.0)
        interPos.x = k * l + point1.x
        interPos.y = k * m + point1.y
        interPos.z = k * n + point1.z

        return interPos


class Wall:
    """
    Class defining a wall plane by 3 points
    """

    def __init__(self, posA: Point, posB: Point, posC: Point):
        self.plane_coeffs = Plane(posA, posB, posC)
        self.IMS = None
        self.node_positions = None
        # Store wall boundaries
        self.corners = [posA, posB, posC]
        # Calculate wall dimensions
        self.width = posB.getDistance(posA)
        self.height = posC.getDistance(posA)

    def is_point_within_bounds(self, point: Point) -> bool:
        """Check if a point lies within the wall boundaries"""
        # Get vectors from first corner to other corners
        v1 = self.corners[1].subtract(self.corners[0])  # width vector
        v2 = self.corners[2].subtract(self.corners[0])  # height vector
        
        # Get vector from first corner to point
        vp = np.array([point.x - self.corners[0].x,
                       point.y - self.corners[0].y,
                       point.z - self.corners[0].z])
        
        # Project vp onto v1 and v2
        proj1 = np.dot(vp, v1) / np.dot(v1, v1)
        proj2 = np.dot(vp, v2) / np.dot(v2, v2)
        
        # Point is within bounds if both projections are between 0 and 1
        return (0 <= proj1 <= 1) and (0 <= proj2 <= 1)

    @property
    def node_positions(self):
        return self._node_positions

    @node_positions.setter
    def node_positions(self, pos):
        self._node_positions = pos


class Point:
    """
    Class that defines a point in 3D cartesian coordinate system
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def getDistance(self, p):
        """
        Returns the Euclidean distance between two 3D positions in
        cartesian coordinates
        """
        return np.sqrt((self.x - p.x) ** 2 + (self.y - p.y) ** 2 + (self.z - p.z) ** 2)

    def subtract(self, p):
        """Returns the vector difference as numpy array"""
        return np.array([self.x - p.x, self.y - p.y, self.z - p.z], dtype=float)

    def equals(self, p):
        if (self.x == p.x) and (self.y == p.y) and (self.z == p.z):
            return True
        else:
            return False

    def to_array(self):
        """Explicit conversion to numpy array"""
        return np.array([self.x, self.y, self.z], dtype=float)


class Plane:
    """
    Class and helper functions defining a 3D plane
    """

    def __init__(self, posA, posB, posC):
        # plane represented by ax + by + cz + d = 0 and its normal vector
        # posA, posB and posC are 3 points on a plane

        # find vector normal to the plane
        arr1 = posB.subtract(posA)
        arr2 = posC.subtract(posA)
        self.normal = np.cross(arr1, arr2)

        assert np.dot(self.normal, arr1) == 0.0, "normal vector not right"

        # scalar component
        self.d = np.dot(-self.normal, [posA.x, posA.y, posA.z])
        self.a = self.normal[0]
        self.b = self.normal[1]
        self.c = self.normal[2]

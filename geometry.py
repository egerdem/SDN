from __future__ import annotations  # This allows forward references in type hints
import numpy as np
from typing import Dict, List


class Room:
    """
    Class defining a room with some properties that can be controlled
    """

    def __init__(self, lx, ly, lz):
        self.label = "cuboid"
        self.nWalls = 6
        self.x = lx
        self.y = ly
        self.z = lz
        self.wallAttenuation = []  # this is a list
        self.wallFilters = dict()  # this is a dictionary
        self._setup_walls()

    def _setup_walls(self):
        # Setting up walls (internal use only)
        self.walls = {
        'south': Wall(Point(0, 0, 0), Point(self.x, 0, 0), Point(0, 0, self.z)),
        'north': Wall(Point(0, self.y, 0), Point(0, self.y, self.z), Point(self.x, self.y, 0)),
        'west' : Wall(Point(0, 0, 0), Point(0, self.y, 0), Point(0, 0, self.z)),
        'east' : Wall(Point(self.x, 0, 0), Point(self.x, self.y, 0), Point(self.x, self.y, self.z)),
        'ceiling' : Wall(Point(0, 0, self.z), Point(0, self.y, self.z), Point(self.x, self.y, self.z)),
        'floor' : Wall(Point(0, 0, 0), Point(0, self.y, 0), Point(self.x, self.y, 0))}

    def set_microphone(self, mx, my, mz):
        self.mx = mx
        self.my = my
        self.mz = mz
        self.micPos = Point(mx, my, mz)

    def set_source(self, sx, sy, sz, signal, Fs=44100):
        self.source = Source(sx, sy, sz, signal, Fs)
        self.srcPos = Point(sx, sy, sz)
        # Calculate SDN node positions (first-order reflection points)
        self._calculate_sdn_nodes()

    def _calculate_sdn_nodes(self):
        """Calculate fixed SDN node positions (first-order reflection points)"""
        for wall_label, wall in self.walls.items():
            # Calculate image source for "wall"
            img_source = ImageSource({wall_label: wall}, self.srcPos, self.micPos)
            # Find intersection point between IM-mic line segment and the wall
            wall.node_positions = img_source._find_intersection_point(img_source.imageSourcePos, img_source.micPos)


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
        self.imageSourcePos = None  # Will be set by get_first_order_image
        self._findImageSources()  # Calculate image sources during initialization

    def _findImageSources(self):
        for wall_label, wall in self.walls.items():
            self.imageSourcePos = self.get_first_order_image(wall)
            wall.IMS = self.imageSourcePos

    def get_first_order_image(self, wall):
        # find image source locations along the plane
        self.d = wall.plane_coeffs.d
        self.a = wall.plane_coeffs.a
        self.b = wall.plane_coeffs.b
        self.c = wall.plane_coeffs.c

        # Compute the distance from the point to the plane
        # Distance formula: (ax + by + cz + d) / sqrt(a^2 + b^2 + c^2)
        norm = self.a ** 2 + self.b ** 2 + self.c ** 2
        dist_to_plane = (self.a * self.srcPos.x + self.b * self.srcPos.y + 
                        self.c * self.srcPos.z + self.d) / norm

        # Compute the reflection point
        self.ImageSource_Pos = Point(0.0, 0.0, 0.0)
        self.ImageSource_Pos.x = self.srcPos.x - 2 * dist_to_plane * self.a
        self.ImageSource_Pos.y = self.srcPos.y - 2 * dist_to_plane * self.b
        self.ImageSource_Pos.z = self.srcPos.z - 2 * dist_to_plane * self.c

        return self.ImageSource_Pos

    def _find_intersection_point(self, point1: Point, point2: Point) -> Point:
        """Find intersection point between IM-mic line segment and wall plane."""
        # Get first wall's plane coefficients (we only have one wall)
        wall = next(iter(self.walls.values()))
        
        # Get direction vector of the line
        l = point2.x - point1.x
        m = point2.y - point1.y
        n = point2.z - point1.z

        # Calculate intersection parameter k
        k = -(wall.plane_coeffs.a * point1.x + 
              wall.plane_coeffs.b * point1.y + 
              wall.plane_coeffs.c * point1.z +
              wall.plane_coeffs.d) / (wall.plane_coeffs.a * l + 
                                    wall.plane_coeffs.b * m + 
                                    wall.plane_coeffs.c * n)

        # Calculate intersection point
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
        self.node_positions = None  # simple attribute instead of property
        # Store wall boundaries
        self.corners = [posA, posB, posC]
        # Calculate wall dimensions


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

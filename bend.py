class Bend:
    def __init__(self,
                 device_id=None,           # str
                 index_start=-1,           # int
                 index_end=-1,             # int
                 time_start=-1,            # UTC
                 time_end=-1,              # UTC
                 axis_y_is_vertical=None,  # bool, shows either y is more like g (True) or x (False)
                 time_down=-1,             # sec
                 time_static=-1,           # sec
                 time_up=-1,               # sec
                 bend_angle=0,             # float
                 roll_angle=0,             # float
                 weight_down=None,         # NOT REQUIRED, FOR PREDICTION PURPOSES ONLY
                 weight_up=None,           # NOT REQUIRED, FOR PREDICTION PURPOSES ONLY
                 pitches=None,             # pitches during the bend
                 acceleration=None,         # acceleration - 1g
                 x=None,
                 y=None,
                 z=None
                 ):

        self.device_id = device_id
        self.index_start = index_start
        self.index_end = index_end
        self.time_start = time_start
        self.time_end = time_end
        self.axis_y_is_vertical = axis_y_is_vertical
        self.time_down = time_down
        self.time_static = time_static
        self.time_up = time_up
        self.bend_angle = bend_angle
        self.roll_angle = roll_angle
        self.weight_down = weight_down
        self.weight_up = weight_up
        self.pitches = pitches
        self.acceleration = acceleration
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return '(bend) down: {d}, static: {st}, up: {up}, bend: {bend}, roll: {roll}'.format(d=self.time_down,
                                                                                             st=self.time_static,
                                                                                             up=self.time_up,
                                                                                             bend=self.bend_angle,
                                                                                             roll=self.roll_angle)

    def __str__(self):
        return self.__repr__()

    def print_times(self):
        print ('{:.0f} - {:.0f}'.format(self.time_start, self.time_end))

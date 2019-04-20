from dynamixel_sdk import PortHandler, PacketHandler
import numpy as np
from hardware.Sim2Robot import toRobot, fromRobot
from hardware.WriteData import SyncMultiWriter
from hardware.ReadData import SyncMultiReader
from hardware.dabs import MultiReader
import hardware.p1mx28
import hardware.p2mx28

P1 = (PacketHandler(1.0), hardware.p1mx28)
P2 = (PacketHandler(2.0), hardware.p2mx28)

# Four of the motors, and the IMU, only support version 1, apparently.
dxl_ids_v1 = [1,2,19,20]
dxl_ids_v2 = list(range(3,19))
dxl_ids_legs = list(range(7,19))
dxl_ids_hip_pitch = [11,12]

def raw_attr(attr, protocol):
    return [(getattr(protocol, 'ADDR_'+attr), getattr(protocol, 'LEN_'+attr))]

class DarwinInterface():
    def __init__(self, init_state):
        self.init_state = init_state
        # Set up the port
        BAUD = 1000000
        self.port_handler = PortHandler("/dev/ttyUSB0")
        if not self.port_handler.openPort():
                raise RuntimeError("Couldn't open port")
        if not self.port_handler.setBaudRate(BAUD):
                raise RuntimeError("Couldn't change baud rate")
        print("############## Successfully Opened Port ######################")
        self.enable_torque()
        self.reset(self.init_state)
        input('Press Enter to continue')
        self.leg_writer = self.build_writer('GOAL_POSITION', P2, dxl_ids_legs)
        # 40 and 42 are pitch and roll
        self.imu_reader = MultiReader(self.port_handler, P1[0], [200], [(40,2),(42,2)])
        self.hip_reader = SyncMultiReader(self.port_handler, P2[0],
                dxl_ids_hip_pitch, raw_attr('PRESENT_POSITION', P2[1]))

        # For integrating / finite-differencing raw measurements.
        self.pitch = 0.
        self.roll = 0.
        self.prev_pitch_rate = 0.
        self.prev_roll_rate = 0.
        self.prev_hip_left = 0. # TODO this should be set based on init_state
        self.prev_hip_right = 0.

    def build_writer(self, attr, p, dxl_ids):
        return SyncMultiWriter(self.port_handler, p[0], dxl_ids, raw_attr(attr, p[1]))

    def enable_torque(self):
        w1_p = self.build_writer('P_GAIN', P1, dxl_ids_v1)
        w1_d = self.build_writer('P_GAIN', P1, dxl_ids_v1)
        w1_TE = self.build_writer('TORQUE_ENABLE', P1, dxl_ids_v1)

        w2_p = self.build_writer('POSITION_P_GAIN', P2, dxl_ids_v2)
        w2_d = self.build_writer('POSITION_D_GAIN', P2, dxl_ids_v2)
        w2_TE = self.build_writer('TORQUE_ENABLE', P2, dxl_ids_v2)
        try:
            # Gains can only be changed while torque is disabled.
            w1_p.write([32]*4)
            w1_d.write([128]*4)
            w2_p.write([255]*16)
            w2_d.write([18]*16)

            # Apparently this is not necessary for P1
            # (motors automatically enable torque when a goal position is set)
            w1_TE.write([1]*4)
            w2_TE.write([1]*16)
        except:
            self.port_handler.closePort()
            raise

    def reset(self, init_state):
        raw_state = toRobot(init_state[6:])
        w1_GP = self.build_writer('GOAL_POSITION', P1, dxl_ids_v1)
        w2_GP = self.build_writer('GOAL_POSITION', P2, dxl_ids_v2)
        try:
            w1_GP.write(raw_state[:2] + raw_state[-2:])
            w2_GP.write(raw_state[2:-2])
        except:
            self.port_handler.closePort()
            raise

    def integrate_imu(self, t0, t1):
        try:
            pitch_rate, roll_rate = self.imu_reader.read()
        except:
            self.port_handler.closePort()
            raise
        robot_to_radian = 1000./1024*np.pi/180*3.5
        pitch_rate = (pitch_rate-512)*robot_to_radian
        roll_rate = -(roll_rate-512)*robot_to_radian
        dt = t1 - t0
        # Trapezoidal integration
        if dt > 1./300: # HACK: sometimes we got absurdly big values on the initial timestep
            self.pitch += (pitch_rate + self.prev_pitch_rate)*dt/2.
            self.roll  += (roll_rate + self.prev_roll_rate)*dt/2.
        self.prev_pitch_rate, self.prev_roll_rate = pitch_rate, roll_rate

    def read(self, t0, t1):
        try:
            hip_right, hip_left = self.hip_reader.read()
        except:
            self.port_handler.closePort()
            raise
        if hip_right == 0: # TODO some weird interference bug makes this happen sometimes
            hip_right = self.prev_hip_right
            hip_left = self.prev_hip_left
        dt = t1 - t0
        # Finite difference to find hip angle rates.
        hip_rate_right = (hip_right - self.prev_hip_right) / dt
        hip_rate_left  = (hip_left - self.prev_hip_left) / dt
        self.prev_hip_right, self.prev_hip_left = hip_right, hip_left

        positions = np.zeros(20, dtype=int)
        positions[10:12] = [self.prev_hip_right, self.prev_hip_left]
        d_positions = np.zeros(20, dtype=int)
        # HACK because fromRobot thinks 2048 is zero.
        d_positions[10:12] = [hip_rate_right+2048, hip_rate_left+2048]
        pose = np.zeros(6)
        pose[4:6] = [self.pitch, self.roll]
        d_pose = np.zeros(6)
        d_pose[4:6] = [self.prev_pitch_rate, self.prev_roll_rate]
        q = np.concatenate([pose, fromRobot(positions)])
        dq = np.concatenate([d_pose, fromRobot(d_positions)])
        # TODO we might have to fix some signs.
        return q, dq

    def write(self, target_q):
        raw_state = toRobot(target_q[6:])
        try:
            self.leg_writer.write(raw_state[6:-2])
        except:
            self.port_handler.closePort()
            raise

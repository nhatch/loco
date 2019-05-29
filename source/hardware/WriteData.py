from dynamixel_sdk import *
import itertools
import time
import numpy as np


class SyncMultiWriter:

    """
    Write to the same contiguous block of memory across multiple motors

    WARNING: Using this with non-contiguous attributes will write anything
    that happens to be in between with 0
    """

    def __init__(self, port_handler, packet_handler, motor_ids, attrs):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids
        self.attrs = attrs

        self.block_start = min(self.attrs)[0]
        self.block_end = sum(max(self.attrs))
        self.block_len = self.block_end - self.block_start

        self.packet = self.construct_packet()

    def construct_packet(self):

        packet = GroupSyncWrite(self.port_handler, self.packet_handler,
                                     self.block_start, self.block_len)

        for motor_id in self.motor_ids:
            if not packet.addParam(motor_id,
                                   [0] * self.block_len):
                raise RuntimeError("Couldn't add any storage for motor %i, " \
                                   + "param %i" % motor_id)

        return packet


    def write(self, targets):

        self.packet.clearParam()
        #print("here")

        for motor_index, motor_id in enumerate(self.motor_ids):
            motor_data = [0] * self.block_len
            motor_targets = targets[motor_index * len(self.attrs)
                                    :(motor_index + 1) * len(self.attrs)]

            for attr_index in range(len(self.attrs)):
                attr = self.attrs[attr_index]

                # Replace the relevant subrange in the data array with the
                # byte list
                motor_data[(attr[0] - self.block_start)
                           :sum(attr) - self.block_start] = \
                                            list(motor_targets[attr_index]
                                                 .to_bytes(attr[1], "little"))

            if not self.packet.addParam(motor_id, motor_data):
                raise RuntimeError("Couldn't set value for motor %i"
                                   % motor_id)

        dxl_comm_result = self.packet.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError("%s"
                               % self.packet_handler.getTxRxResult( \
                                                            dxl_comm_result))
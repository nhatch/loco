from dynamixel_sdk import *
import itertools
import time
import numpy as np




class SyncMultiReader:

    """
    Write to the same contiguous block of memory across multiple motors

    WARNING: Using this with non-contiguous attributes will write anything
    that happens to be in between with 0
    """

    def __init__(self, port_handler, packet_handler, motor_ids, attrs):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids
        self.attrs = attrs[0]

        #self.block_start = min(self.attrs)[0]
        #self.block_end = sum(max(self.attrs))
        #self.block_len = self.block_end - self.block_start

        self.packet = self.construct_packet()
        

    def construct_packet(self):

        packet = GroupSyncRead(self.port_handler, self.packet_handler,
                                     self.attrs[0], self.attrs[1])

        for motor_id in self.motor_ids:
            
            if not packet.addParam(motor_id,):
                raise RuntimeError("Couldn't add any storage for motor %i, " \
                                   + "param %i" % motor_id)
   
        return packet


    def read(self, ):

        
        comm_result = self.packet.txRxPacket()
        
        result = []
        for motor_index, motor_id in enumerate(self.motor_ids):
            result.append(self.packet.getData(motor_id,self.attrs[0],self.attrs[1]))
            
        return result


class BulkMultiReader():

    """
    Read multiple attributes via BulkRead
    """

    def __init__(self, port_handler, packet_handler, motor_ids, attrs):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids

        self.attrs = attrs[0]

        # Python compares tuples by entry going left to right, so the
        # following logic works
        #self.block_start = min(self.attrs)[0]
        #self.block_end = sum(max(self.attrs))
        #self.block_len = self.block_end - self.block_start

        self.packet = self.construct_packet()

    def construct_packet(self):

        packet = GroupBulkRead(self.port_handler, self.packet_handler)

        for motor_id in self.motor_ids:
            if not packet.addParam(motor_id, self.attrs[0],self.attrs[1]):
                raise RuntimeError("Couldn't add parameter for motor %i",
                                   motor_id)

        return packet

    def read(self):

        results = [None] * (len(self.motor_ids) * len(self.attrs))

        comm_result = self.packet.txRxPacket()
        #if comm_result != COMM_SUCCESS:
        #    raise RuntimeError(self.packet_handler.getTxRxResult(comm_result))

        for motor_index, motor_id in enumerate(self.motor_ids):
            for attr_index, attr in enumerate(self.attrs):
                if not self.packet.isAvailable(motor_id, self.attrs[0],self.attrs[1]):
                    raise RuntimeError("Data unavailable for " + str(motor_id)
                                       + ", attribute " + str(attr))

                data_location = len(self.attrs) * motor_index + attr_index
                results[data_location] = self.packet.getData(
                    motor_id, self.attrs[0],self.attrs[1])

        return results

class BulkMultiReaderIMU():

    """
    Read multiple attributes via BulkRead
    """

    def __init__(self, port_handler, packet_handler, motor_ids, attrs):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids

        self.attrs = attrs

        # Python compares tuples by entry going left to right, so the
        # following logic works
        self.block_start = min(self.attrs)[0]
        self.block_end = sum(max(self.attrs))
        self.block_len = self.block_end - self.block_start

        self.packet = self.construct_packet()

    def construct_packet(self):

        packet = GroupBulkRead(self.port_handler, self.packet_handler)

        for motor_id in self.motor_ids:
            if not packet.addParam(motor_id, self.block_start,self.block_len):
                raise RuntimeError("Couldn't add parameter for motor %i",
                                   motor_id)

        return packet

    def read(self):

        results = [None] * (len(self.motor_ids) * len(self.attrs))

        comm_result = self.packet.txRxPacket()
        #if comm_result != COMM_SUCCESS:
        #    raise RuntimeError(self.packet_handler.getTxRxResult(comm_result))

        for motor_index, motor_id in enumerate(self.motor_ids):
            for attr_index, attr in enumerate(self.attrs):
                if not self.packet.isAvailable(motor_id, *attr):
                    raise RuntimeError("Data unavailable for " + str(motor_id)
                                       + ", attribute " + str(attr))

                data_location = len(self.attrs) * motor_index + attr_index
                results[data_location] = self.packet.getData(
                    motor_id, *attr)

        return results
# Import dxl stuff up here

from dynamixel_sdk import *
import itertools
import numpy as np
import joblib
def setup_indirects(self, port_handler, packet_handler, motor_ids, attrs,
                    indirect_root):
    """
    Given appropriate port constructs, list of motor ids, attributes, and
    an indirect index (from 0-55 on the mx28), set the indirect addresses
    on each mx28 so that the (indirected) attributes form a contiguous
    block of memory suitable for sync/bulk rw

    TODO There are two blocks of indirect addresses/data, each with a capacity
    of 28. This function ONLY sets up one contiguous block. It does not
    attempt to detect points where it should jump blocks and do so, or even
    fail when it should

    WARNING/TODO: Apparently, indirect addresses cannot be set while motor
    torque is enabled. Therefore, this function DISABLES TORQUE on all motors
    and makes no attempt to restore it to those motors which were enabled

    WARNING: This function is specifically designed for MX28 right now
    This link should clear up all the "magic numbers"
    http://emanual.robotis.com/docs/en/dxl/mx/mx-28/#control-table-data-address
    """

    # Array which will eventually store list of attributes in form
    # [(indirect_attr_1_address, attr1_len), ...]
    indirected_attrs = [None]

    # There are two blocks of indirect addresses/data, so it takes
    # a bit of logic to set the starting points right based on the indices
    # TODO Validate data lengths to make sure we have enough space given
    # the current attributes and indirect root
    indirect_addr = None
    data_addr = None
    if indirect_root <= 27:
        curr_addr = 2 * indirect_root + 168
        data_addr = 224 + indirect_root
    else:
        curr_addr = 2 * (indirect_root - 27) + 578
        data_addr = 634 + (indirect_root - 27)

    zero_torques()

    # Calculate and set the indirect addresses appropriately
    for attr_index, attr in enumerate(attrs):

        indirected_attrs[attr_index] = (data_addr, attr[1])
        data_addr += attr[1]

        # Addresses / attributes may span multiple words, so there's a tiny bit
        # of logic to handle that correctly; ie if an attr spans four words,
        # make sure all four of those are being mirrored
        for offset in range(attr[1]):
            for dxl_id in motor_ids:

                dxl_comm_result,
                dxl_error = packet_handler.write2ByteTxRx(port_handler,
                                                          dxl_id,
                                                          indirect_addr,
                                                          attr[0] + offset)

                if dxl_comm_result != COMM_SUCCESS:
                    raise RuntimeError("Communication error on setting " \
                                       + "mtor %i's address %i:\n%s"
                                       % dxl_id, attr[0],
                                       packet_handler.getTxRxResult(
                                           dxl_comm_result))
                elif dxl_error != 0:
                    raise RuntimeError("Hardware error on setting motor %i's " \
                                       + "address %i:\n%s" %
                                       dxl_id, attr[0],
                                       packet_handler.getRxPacketError
                                       (dxl_error))

            # Each address is two bytes, as there are more than 256
            indirect_addr += 2

    return indirected_attrs


def zero_torques(port_handler, packet_handler, motor_ids):

    for dxl_id in self.motors_ids:

        # TODO Specific to MX28
        comm_result, error = self.packet_handler.write1ByteTxRx(
            self.port_handler, dxl_id, mx28.ADDR_TORQUE_ENABLE, 0)

        if comm_result != COMM_SUCCESS:
            raise RuntimeError("Comm error on trying to disable motor %i:\n%s"
                               % dxl_id,
                               self.packet_handler.getTxRxResult(comm_result))
        elif error != 0:
            raise RuntimeError("Hardware error on disabling motor %i:\n%s"
                               % dxl_id,
                               self.packet_handler.getRxPacketError(error))


class BulkMultiReader():

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
            if not packet.addParam(motor_id, 36,2):
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
                if not self.packet.isAvailable(motor_id, 36,2):
                    raise RuntimeError("Data unavailable for " + str(motor_id)
                                       + ", attribute " + str(attr))

                data_location = len(self.attrs) * motor_index + attr_index
                results[data_location] = self.packet.getData(
                    motor_id, 36,2)

        return results

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

class SyncMultiIMUReader:

    

    def __init__(self, port_handler, packet_handler, motor_ids, attrs):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids
        self.attrs = attrs

        self.block_start = 38#min(self.attrs)[0]
        self.block_end = sum(max(self.attrs))
        self.block_len = 12#self.block_end - self.block_start

        self.packet = self.construct_packet()
        print("packet",self.packet)

    def construct_packet(self):

        packet = GroupSyncRead(self.port_handler, self.packet_handler,
                                     38, 2)

        for motor_id in self.motor_ids:
            #print("motor added")
            if not packet.addParam(motor_id,):
                raise RuntimeError("Couldn't add any storage for motor %i, " \
                                   + "param %i" % motor_id)

        print("dictionary",packet.data_dict)
        return packet


    def read(self, ):

        #self.packet.clearParam()
        #print("data dict",self.packet.data_dict)
        comm_result = self.packet.txRxPacket()
        print("data dict",comm_result)
        result = []
        for motor_index, motor_id in enumerate(self.motor_ids):
            #print("is isAvailable",self.packet.isAvailable(motor_id,132,4))
            result.append(self.packet.getData(motor_id,38,2))
            print("result",result)
            #motor_data = [0] * self.block_len
            #motor_targets = targets[motor_index * len(self.attrs)
            #                        :(motor_index + 1) * len(self.attrs)]

            #for attr_index in range(len(self.attrs)):
            #    attr = self.attrs[attr_index]

                # Replace the relevant subrange in the data array with the
                # byte list
            #    motor_data[(attr[0] - self.block_start)
            #               :sum(attr) - self.block_start] = \
            #                                list(motor_targets[attr_index]
            #                                     .to_bytes(attr[1], "little"))

            #if not self.packet.addParam(motor_id):
            #    raise RuntimeError("Couldn't read value for motor %i"
            #                       % motor_id)
        print("actual packet",self.packet.data_dict[200])
        #dxl_comm_result = self.packet.rxPacket()
        #if dxl_comm_result != COMM_SUCCESS:
        #    raise RuntimeError("%s"
        #                       % "Error")
        return result




class SyncMultiReader:

    

    def __init__(self, port_handler, packet_handler, motor_ids, attrs):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids
        self.attrs = attrs

        self.block_start = 132#min(self.attrs)[0]
        self.block_end = sum(max(self.attrs))
        self.block_len = 4#self.block_end - self.block_start

        self.packet = self.construct_packet()
        print("packet",self.packet)

    def construct_packet(self):

        packet = GroupSyncRead(self.port_handler, self.packet_handler,
                                     38, 2)

        for motor_id in self.motor_ids:
            #print("motor added")
            if not packet.addParam(motor_id,):
                raise RuntimeError("Couldn't add any storage for motor %i, " \
                                   + "param %i" % motor_id)

        print("dictionary",packet.data_dict)
        return packet


    def read(self, ):

        #self.packet.clearParam()
        #print("data dict",self.packet.data_dict)
        comm_result = self.packet.txRxPacket()
        #print("data dict",comm_result)
        result = []
        for motor_index, motor_id in enumerate(self.motor_ids):
            #print("is isAvailable",self.packet.isAvailable(motor_id,132,4))
            result.append(self.packet.getData(motor_id,38,2))
            #print("result",result)
            #motor_data = [0] * self.block_len
            #motor_targets = targets[motor_index * len(self.attrs)
            #                        :(motor_index + 1) * len(self.attrs)]

            #for attr_index in range(len(self.attrs)):
            #    attr = self.attrs[attr_index]

                # Replace the relevant subrange in the data array with the
                # byte list
            #    motor_data[(attr[0] - self.block_start)
            #               :sum(attr) - self.block_start] = \
            #                                list(motor_targets[attr_index]
            #                                     .to_bytes(attr[1], "little"))

            #if not self.packet.addParam(motor_id):
            #    raise RuntimeError("Couldn't read value for motor %i"
            #                       % motor_id)

        #dxl_comm_result = self.packet.rxPacket()
        #if dxl_comm_result != COMM_SUCCESS:
        #    raise RuntimeError("%s"
        #                       % "Error")
        return result

class MultiReader():

    def __init__(self, port_handler, packet_handler, motor_ids, attrs):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids
        self.attrs = attrs
        self.protocol_version = 1
        self.identified_attrs = list(itertools.product(self.motor_ids,
                                                       self.attrs))

    def read(self):


        results = [None] * len(self.identified_attrs)

        for index, (id, attr) in enumerate(self.identified_attrs):

            if attr[1] == 1:
                #print("herererer")
                val, comm, err = \
                        self.packet_handler.read1ByteTxRx(self.port_handler,
                                                            #self.protocol_version,
                                                          id, attr[0])
            elif attr[1] == 2:
                val, comm, err = \
                        self.packet_handler.read2ByteTxRx(self.port_handler,
                                                          id, attr[0])
                #print(comm)
            elif attr[1] == 4:
                val, comm, err = \
                        self.packet_handler.read4ByteTxRx(self.port_handler,
                                                          #self.protocol_version,
                                                          id, attr[0])
            else:
                raise RuntimeError("Invalid data size")

            if comm != COMM_SUCCESS:
                raise RuntimeError("Comm error on reading motor %i, " \
                                   + "attribute %s"% id, attr)
            elif err != 0:
                raise RuntimeError("Hardware error on reading motor %i, " \
                                   + "attribute %s" % id, attr)

            results[index] = val
        return results

class MultiWriter:

    def __init__(self, port_handler, packet_handler, motor_ids, attrs):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids
        self.attrs = attrs

        self.identified_attrs = list(itertools.product(self.motor_ids,
                                                       self.attrs))

    def write(self, targets):

        for index, (id, attr) in enumerate(self.identified_attrs):
            if attr[1] == 1:
                comm, err = \
                    self.packet_handler.write1ByteTxRx(self.port_handler, id,
                                                       attr[0], targets[index])
            elif attr[1] == 2:
                comm, err = \
                    self.packet_handler.write2ByteTxRx(self.port_handler, id,
                                                       attr[0], targets[index])
            elif attr[1] == 4:
                comm, err = \
                    self.packet_handler.write4ByteTxRx(self.port_handler, id,
                                                       attr[0], targets[index])
                #print("comm",comm)
            else:
                raise RuntimeError("Invalid data size")

            if comm != COMM_SUCCESS:
                raise RuntimeError("Comm error on writing motor %i," \
                                   + "attribute %s" % id, attr)
            elif err != 0:
                raise RuntimeError("Hardware error on writing motor %i, " \
                                   + "attribute %s" % id, attr)
class KalmanPosition():

    def __init__(self,):
        self.P = np.diag(100000*np.ones(9,))
        self.A = np.diag(np.ones(9))
        self.Q = np.diag(np.array([0,0,0,0,0,0,0,0,0]))
        
        self.x = np.zeros(9,)
        self.R = np.diag(0.0001*np.array([1e-2,1e-2,1e-2,1e-2,1e-2,1e-2,0.2**2,0.26**2,0.40**2]))
        

    def filter(self,measurement,dt):
        
        self.A[0,3] = dt
        self.A[1,4] = dt
        self.A[2,5] = dt

        self.A[0,6] = 0.5*dt**2
        self.A[1,7] = 0.5*dt**2
        self.A[2,8] = 0.5*dt**2

        self.A[3,6] = dt
        self.A[4,7] = dt
        self.A[5,8] = dt

        X_predicted = np.dot(self.A,self.x)
        #print(X_predicted.shape)
        P_predicted = np.matmul(np.matmul(self.A,self.P),np.transpose(self.A)) + self.Q
        #print(P_predicted.shape)
        
        H = np.diag(np.array([0,0,0,0,0,0,1,1,1]))

        y = measurement - np.dot(H,self.x)
        #print(y)
        
        S = np.matmul(np.matmul(H,P_predicted),np.transpose(H)) + self.R

        K = np.matmul(np.matmul(P_predicted,np.transpose(H)),np.linalg.inv(S))
        #print(K)
        self.x = X_predicted + np.dot(K,y)
        #print(self.x - X_predicted)
        ikh = (np.diag(np.ones(9)) - np.matmul(K,H))
        self.P = np.matmul(np.matmul(ikh,P_predicted),np.transpose(ikh)) + np.matmul(np.matmul(K,self.R),np.transpose(K))

        return self.x


def ConvertReadings(imu):
    real2sim_gyro = 1001/1024
    real2sim_acc = 9/1024

    gyro = (imu[:3] - 512)*real2sim_gyro

    acc = imu[3:]
    acc[:2] -= 512
    acc[-1] -= 512+ 1/real2sim_acc
    for i in range(3):
        acc[i]*=9.81*real2sim_acc   
    #print(gyro[0])
    
    #acc*=9.81*real2sim_acc
    #print(acc[0])
    imu_converted = np.array([gyro[0],gyro[1],gyro[2],acc[0],acc[1],acc[2]])
    #print(imu_converted)
    return imu_converted

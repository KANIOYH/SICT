#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件    :hamming.py\n
@说明    :基于hamming(7,4)编码模块\n
@时间    :2022/03/12 21:57:52\n
@作者    : SICT 制作委员会\n
@版本    :1.0\n
'''


import numpy as np
 
class Hamming:
    '''
    Hamming (7,4) error correction code implementation.
    Can be used to encode, parity check, error correct, decode and get the orginal message back.
    This can detect two bit errors and correct single bit errors.
    '''
    _gT = np.matrix([[1, 1, 0, 1], [1, 0, 1, 1], [1, 0, 0, 0], [
        0, 1, 1, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    _h = np.matrix(
        [[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]])

    _R = np.matrix([[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [
        0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])

    def _strToMat(self, binaryStr):
        '''
        @Input
        Binary string of length 4
        
        @Output
        Numpy row vector of length 4
        '''

        inp = np.frombuffer(binaryStr.encode(), dtype=np.uint8) - ord('0')
        return inp

    def encode(self, message):
        '''
        @Input
        String
        Message is a 4 bit binary string
        
        @Output
        Numpy matrix column vector
        Encoded 7 bit binary string
        '''
        message = np.matrix(self._strToMat(message)).transpose()
        en = np.dot(self._gT, message) % 2
        return en

    def parityCheck(self, message):
        '''
        @Input
        Numpy matrix a column vector of length 7
        Accepts a binary column vector
        
        @Output
        Numpy row vector of length 3
        Returns the single bit error location as row vector
        '''
        z = np.dot(self._h, message) % 2
        return np.fliplr(z.transpose())

    def getOriginalMessage(self, message):
        '''
        @Input
        Numpy matrix a column vector of length 7
        Accepts a binary column vector
        
        @Output
        List of length 4
        Returns the single bit error location as row vector ()
        '''
        ep = self.parityCheck(message)
        pos = self._binatodeci(ep)
        if pos > 0:
            correctMessage = self._flipbit(message, pos)
        else:
            correctMessage = message

        origMessage = np.dot(self._R, correctMessage)
        return origMessage.transpose().tolist()

    def _flipbit(self, enc, bitpos):
        '''
        @Input
          enc:Numpy matrix a column vector of length 7
          Accepts a binary column vector
          bitpos: Integer value of the position to change
          flip the bit. Value should be on range 1-7
          
        @Output
          Numpy matrix a column vector of length 7
          Returns the bit flipped matrix
        '''
        enc = enc.transpose().tolist()
        bitpos = bitpos - 1
        if (enc[0][bitpos] == 1):
            enc[0][bitpos] = 0
        else:
            enc[0][bitpos] = 1
        return np.matrix(enc).transpose()

    def _binatodeci(self, binaryList):
        '''
        @Input
        Numpy matrix column or row one dimension
        
        @Output
        Decimal number equal to the binary matrix
        '''
        return sum(val * (2 ** idx) for idx, val in enumerate(reversed(binaryList.tolist()[0])))



def encode(msg_base, msg_ext):
    """hamming(7,4)编码方法
    
    对水印信息进行四位一组，然后hamming(7,4)编码

    Parameters
    ----------
    msg_base: tuple
        基本信息（时间、IP）
    msg_ext: str
        用户自定义的额外信息

    Returns
    ----------
    list
        经过编码的水印信息
    """
    ans = []
    msg = ""
    msg_ip = msg_base[0].split('.')
    msg_time = msg_base[1]

    for byte in msg_ip:
        bits = bin(int(byte)).replace('0b', '')
        tmp = '0' * (8 - len(bits))
        bits = tmp + bits
        msg += bits
        # print(bits)

    msg_time = str(bin(msg_time).replace('0b', ''))
    tmp = '0' * (32 - len(msg_time))
    msg_time = tmp + msg_time
    msg += msg_time

    msg_ext = bytes(msg_ext.encode("gbk"))

    for byte in msg_ext:
        bits = bin(byte).replace('0b', '')
        tmp = '0' * (8 - len(bits))
        bits = tmp + bits
        msg += bits

    ham = Hamming()
    for i in range(0, len(msg), 4):
        tmp = ham.encode(msg[i:i + 4]).flatten()
        ans.extend(np.array(tmp)[0])
    return ans


def decode(msg):
    """hamming(7,4)解码方法
    
    对四位一组水印信息进行合并，然后hamming(7,4)解码

    Parameters
    ----------
    msg:list
        待提取的水印二进制序列

    Returns
    ----------
    tuple
        提取出的IP、时间、用户自定义信息
    """

    msg_ip = []
    msg_time = []
    msg_ext = []
    ham = Hamming()
    if np.array(msg).sum() == 0 or np.array(msg).sum() == len(msg):
        return ("null", "null")

    for i in range(0, len(msg) - len(msg) % 14, 14):
        half1 = np.array(msg[i: i + 7]).reshape((7, 1))
        half2 = np.array(msg[i + 7: i + 14]).reshape((7, 1))
        # print(half1)
        if np.array(msg[i: i + 7]).sum() == 0 and np.array(msg[i: i + 7]).sum() == 0 and i >= 112:
            break
        tmp1 = [str(j) for j in ham.getOriginalMessage(half1)[0]]
        tmp2 = [str(j) for j in ham.getOriginalMessage(half2)[0]]
        tmp = int("".join(tmp1 + tmp2), 2)
        if i < 56:
            msg_ip.append(str(tmp))
        elif i < 112:
            msg_time.append("".join(tmp1 + tmp2))
        else:
            msg_ext.append(tmp)

    msg_ip = ".".join(msg_ip)

    msg_time = str(int("".join(msg_time), 2))
    tmp = '0' * (10 - len(msg_time))
    msg_time = tmp + msg_time
    tmp = list("".join(msg_time))
    tmp[0], tmp[1], tmp[2], tmp[3] = tmp[2], tmp[3], tmp[0], tmp[1]
    tmp = "".join(tmp)
    msg_time = "20" + tmp[:2] + "/" + str(int(tmp[2:4])) + "/" + str(int(tmp[4:6])) + " " + tmp[6:8] + ":" + tmp[8:]

    try:
        if len(msg_ext) > 0:
            msg_ext = bytes(msg_ext).decode("gbk")
    except UnicodeDecodeError:
        return ("fail", "fail")

    return (msg_ip, msg_time, msg_ext)

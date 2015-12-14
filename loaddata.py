import os
import sys
from smap.archiver.client import SmapClient
from smap.contrib import dtutil

class SMAPDataRetriever(object):
    def __init__(self):
        self.smapc = SmapClient("http://new.openbms.org/backend")

    def get_data(self, room, building, startDate, endDate):
        print "Pulling in data for room : ", room

        data_temp = self.smapc.query("apply window(mean, field='minute',width=10) to data in ('%s','%s') where Metadata/Name='STA_%s____ART'" % (startDate, endDate, room))
        data_reheat = self.smapc.query("apply window(mean, field='minute',width=10) to data in ('%s','%s') where Metadata/Name='STA_%s____RVP'" % (startDate, endDate, room))
        data_flow = self.smapc.query("apply window(mean, field='minute',width=10) to data in ('%s','%s') where Metadata/Name='STA_%s___SVEL'" % (startDate, endDate, room))
        data_oat = self.smapc.query("apply window(mean, field='minute',width=10) to data in ('%s','%s') where Metadata/Name='STA_%s__OAT'" % (startDate, endDate, building))

        data = {}
        for reading in data_temp[0]["Readings"]:
            data[int(reading[0])] = {}
            data[int(reading[0])]["temp"] = float(reading[1])

        for reading in data_reheat[0]["Readings"]:
            data[int(reading[0])]["reheat"] = float(reading[1])

        for reading in data_flow[0]["Readings"]:
            data[int(reading[0])]["flow"] = float(reading[1])

        for reading in data_oat[0]["Readings"]:
            data[int(reading[0])]["outtemp"] = float(reading[1])
        return data

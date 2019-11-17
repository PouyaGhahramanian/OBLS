import csv
import pandas as pd

Airline_dict = {"US": 0, "AA": 1, "AS": 2, "CO": 3, "DL": 4, "B6": 5, "HA": 6, "OO": 7, "9E": 8, "OH": 9,
                "EV": 10, "XE": 11, "YV": 12, "UA": 13, "MQ": 14, "FL": 15, "F9": 16, "WN": 17}

AirportFrom_list =  ['PHX','LAX','SFO','ANC','LAS','SLC','DEN','ONT','FAI','BQN','PSE','HNL','BIS','IYK','EWR','BOS','MKE','GFK','OMA','GSO','LMT','SEA','MCO','TPA','DLH','MSP','FAR','MFE','MSY','VPS','BWI','MAF','LWS','RST','ALB','DSM','CHS','MSN','JAX','SAT','PNS','BHM','LIT','SAV','BNA','ICT','ECP','DHN','MGM','CAE','PWM','ACV','EKO','PHL','ATL','PDX','RIC','BTR','HRL','MYR','TUS','SBN','CAK','TVC','CLE','ORD','DAY','MFR','BTV','TLH','TYS','DFW','FLL','AUS','CHA','CMH','LRD','BRO','CRP','LAN','PVD','FWA','JFK','LGA','OKC','PIT','PBI','ORF','DCA','AEX','SYR','SHV','VLD','BDL','FAT','BZN','RDM','LFT','IPL','EAU','ERI','BUF','IAH','MCI','AGS','ABI','GRR','LBB','CLT','LEX','MBS','MOD','AMA','SGF','AZO','ABE','SWF','BGM','AVP','FNT','GSP','ATW','ITH','TUL','COS','ELP','ABQ','SMF','STL','IAD','DTW','RDU','RSW','OAK','ROC','IND','CVG','MDW','SDF','ABY','TRI','XNA','ROA','MLI','LYH','EVV','HPN','FAY','EWN','CSG','GPT','MLU','MOB','OAJ','CHO','ILM','BMI','PHF','ACY','JAN','CID','GRK','HOU','CRW','HTS','PSC','BOI','SBP','CLD','PSP','SBA','MEM','MRY','GEG','RDD','PAH','CMX','SPI','EUG','CIC','PIH','SGU','COD','MIA','MHT','GRB','FSD','SJU','AVL','BFL','RAP','DRO','PIA','OGG','SIT','TXK','RNO','DAL','SCE','MEI','MDT','FCA','SJC','KOA','PLN','SAN','GNV','HLN','GJT','CPR','FSM','CMI','GTF','HDN','ITO','MTJ','HSV','BTM','BIL','COU','MSO','SMX','TWF','ISP','GCC','LIH','LNK','DAB','SNA','MQT','LGB','CWA','LSE','BUR','ACT','MHK','MOT','IDA','SUN','GTR','MLB','SRQ','JAC','ASE','LCH','JNU','ROW','BQK','YUM','FLG','EGE','GUC','EYW','RKS','BGR','ELM','ADQ','OTZ','OTH','STT','KTN','BET','SJT','CDC','CEC','SPS','SCC','STX','OME','MKG','WRG','TYR','BRW','GGG','PSG','BKG','YAK','CLL','SAF','CYS','LWB','CDV','FLO','BLI','DBQ','TOL','UTM','PIE','ADK','ABR','TEX','MMH','GUM']

AirportTo_list = ['CLT','DFW','SEA','IAH','MSP','DTW','ORD','ATL','PDX','JFK','SLC','HNL','PHX','MCO','OGG','LAX','KOA','ITO','SFO','MIA','IAD','SMF','PHL','LIH','DEN','LGA','MEM','CVG','YUM','CWA','MKE','BQN','FAI','LAS','ANC','BOS','LGB','FLL','SJU','EWR','DCA','BWI','RDU','MCI','TYS','SAN','ONT','OAK','MDW','BNA','DAL','CLE','JAX','JNU','RNO','ELP','SAT','OTZ','MBS','BDL','STL','HOU','AUS','SNA','SJC','LIT','TUS','TUL','CMH','LAN','IND','AMA','CRP','PIT','RKS','FWA','TPA','PBI','JAN','DSM','ADQ','GRB','PVD','ABQ','SDF','RSW','MSY','BUR','BOI','TLH','BHM','ACV','ORF','BET','KTN','RIC','SRQ','BTR','XNA','MHT','GRR','SBN','SBA','ROA','CID','GPT','MFR','SGU','HPN','OMA','OTH','GSP','LMT','BUF','MSN','BFL','CAE','HRL','OKC','SYR','COS','BTV','CDC','SCC','DAY','SJT','TVC','ROC','ISP','MRY','SBP','MLI','MOB','CIC','SAV','FAT','EKO','GEG','ECP','LFT','SUN','HSV','SHV','CHA','CAK','BZN','MAF','GSO','MDT','PHF','ICT','AZO','RAP','CHS','CLD','MKG','VPS','PIH','ATW','AGS','PNS','BIL','SPI','FAR','CPR','PIA','SPS','TWF','LBB','ALB','CEC','DRO','GJT','GNV','RST','AVL','GRK','PSP','LEX','TRI','SGF','FSM','RDD','OME','MFE','LSE','BMI','MYR','FAY','FSD','EUG','MGM','EVV','MLB','FNT','STT','WRG','ABE','BIS','MOT','MLU','GFK','RDM','COU','LRD','PSC','MOD','PWM','ILM','ABY','CRW','TXK','BRO','BRW','EYW','DAB','ROW','ABI','EAU','TYR','MSO','FLG','CSG','VLD','DHN','OAJ','AEX','CHO','SAF','GGG','FCA','ASE','BKG','MHK','LNK','MQT','YAK','GTR','SMX','SWF','ITH','AVP','ELM','BGM','SIT','PSG','CYS','CLL','SCE','LWB','LCH','GCC','IYK','LWS','COD','HLN','BQK','GTF','DLH','BTM','EGE','IDA','JAC','HDN','MTJ','CMX','CMI','CDV','LYH','ACT','STX','IPL','PAH','HTS','MEI','BLI','ERI','EWN','FLO','ACY','DBQ','TOL','GUC','PLN','BGR','PSE','PIE','UTM','ADK','ABR','TEX','MMH','GUM']

in_from = set(AirportFrom_list)
in_to = set(AirportTo_list)
in_to_but_not_in_from = in_to - in_from
Airport_list = AirportFrom_list + list(in_to_but_not_in_from)

Airport_dict = {Airport_list[i]: i for i in range(0, len(Airport_list) ) }

Dicts = Airline_dict
Dicts.update(Airport_dict)

new_rows = []

i = 0
with open('data/airlines.csv', 'rt') as f:
    reader = csv.reader(f)
    for row in reader:
        new_row = row
        for key, value in Airport_dict.items():
            new_row = [ x.replace(key, str(value)) for x in new_row ]
        for key, value in Airline_dict.items():
            new_row = [x.replace(key, str(value)) for x in new_row]
        new_rows.append(new_row)
        i += 1
        if(i % 100 == 0):
            print(i)

with open('data/airlines_numerical.csv', 'wt') as f:
    writer = csv.writer(f)
    writer.writerows(new_rows)

'''
df=pd.read_csv("data/airlines_num.csv",converters={"Airline":int, "AirportFrom":int, "AirportTo":int})
df.to_csv("data/airlines_numerical.csv")
'''

import datetime, pytz
from datetime import timedelta

tz = pytz.timezone("Europe/Copenhagen")
def to_utc(d):
    return tz.normalize(tz.localize(d)).astimezone(pytz.utc)

def parse_date(date_str, convert_to_utc=True):
    local = pytz.timezone ("Europe/Copenhagen")
    
    if len(date_str)==16:
        #this condition is for bcc
        format_str = '%d%b%y:%H:%M:%S'
    
    elif len(date_str) == 10:
        #this condition is for labka
        format_str = '%d%m%y%H%M'
    
    elif len(date_str) == 19: 
        format_str = '%Y-%m-%d %H:%M:%S'
    else:
        raise Exception ("Data format not recognized", date_str)
    
    if convert_to_utc:
        d = datetime.datetime.strptime(date_str, format_str)
        utc_dt = to_utc(d)
        return utc_dt
    else:
        d = datetime.datetime.strptime(date_str, format_str)
        utc_dt = pytz.utc.localize(d)
        return utc_dt
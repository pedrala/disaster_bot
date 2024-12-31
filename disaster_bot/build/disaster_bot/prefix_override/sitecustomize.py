import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/viator/ws/ttb4_ws/disaster_bot/install/disaster_bot'

from websocket import create_connection
import json
import random
import string
import re

# parse inputs
symbol = "QQQ"
silent = True
exitoninput = True

# Create a connection to the tunnel
ws = create_connection(
    'wss://data.tradingview.com/socket.io/websocket',
    headers=json.dumps({'Origin': 'https://data.tradingview.com'})
)
session = "qs_" + (''.join(random.choice(string.ascii_lowercase) for i in range(12)))
if not silent: print("session generated {}".format(session))
chart_session = "cs_" + (''.join(random.choice(string.ascii_lowercase) for i in range(12)))
if not silent: print("chart_session generated {}".format(chart_session))

# Then send a message through the tunnel
def sendMessage(ws, func, args):
    msg = json.dumps({"m":func, "p":args}, separators=(',', ':'))
    ws.send("~m~" + str(len(msg)) + "~m~" + msg)
sendMessage(ws, "set_auth_token", ["unauthorized_user_token"])
sendMessage(ws, "chart_create_session", [chart_session, ""])
sendMessage(ws, "quote_create_session", [session])
sendMessage(ws, "quote_set_fields", [session,"ch","chp","current_session","description","local_description","language","exchange","fractional","is_tradable","lp","lp_time","minmov","minmove2","original_name","pricescale","pro_name","short_name","type","update_mode","volume","currency_code","rchp","rtc"])
sendMessage(ws, "quote_add_symbols",[session, symbol, {"flags":['force_permission']}])
sendMessage(ws, "quote_fast_symbols", [session,symbol])
sendMessage(ws, "resolve_symbol", [chart_session,"symbol_1","={\"symbol\":\"" + symbol + "\",\"adjustment\":\"splits\",\"session\":\"extended\"}"])
if not silent:
    sendMessage(ws, "create_series", [chart_session, "s1", "s1", "symbol_1", "1", 5000])

# Printing all the results
while True:
    try:
        result = ws.recv()
        if not silent:
            print(result)
        if silent:
            if not re.search(r'"lp":(.*?),', result) is None:
                print(re.search(r'"lp":(.*?),', result).group(1))
                if exitoninput:
                    exit(0)
        with open("scraped_output.txt", "a") as ww:
            ww.write(result)
            ww.close()
    except Exception as e:
        print(e)
        break

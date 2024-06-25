# Import the necessary libraries
import plotly
import plotly.subplots
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# returns True if seq1 crosses under seq2 at the index
def cross_under(seq1, seq2, index):
    if index < 1: return False
    seq1 = np.array(seq1)
    seq2 = np.array(seq2)
    return seq1[index] < seq2[index] and seq1[index-1] >= seq2[index-1]

class StrategyAnalyzer:
    def __init__(self,
                 SYMBOL="TQQQ",
                 INITIAL_CAPITAL=1000000,
                 COMMISSION_PER_SHARE_TRADED=0,
                 MONTHLY_INSTALLMENTS=0,
                 START_DATE=pd.Timestamp("2019-06-17"),
                 END_DATE=pd.Timestamp("2024-06-17"),
                 EMA_FAST=20,
                 EMA_SLOW=60,
                 EMA_SUPERSLOW=120,
                 RSI_THRESHOLD=70,
                 SELL_PERCENTAGE=0.01,
                 TDFI_THRESHOLD=-100,
                 ):
        # Trading parameters
        self.SYMBOL = SYMBOL
        self.INITIAL_CAPITAL = INITIAL_CAPITAL
        self.COMMISSION_PER_SHARE_TRADED = COMMISSION_PER_SHARE_TRADED
        self.MONTHLY_INSTALLMENTS = MONTHLY_INSTALLMENTS
        self.START_DATE = START_DATE
        self.END_DATE = END_DATE

        # Strategy parameters
        self.EMA_FAST = EMA_FAST
        self.EMA_SLOW = EMA_SLOW
        self.EMA_SUPERSLOW = EMA_SUPERSLOW
        self.RSI_THRESHOLD = RSI_THRESHOLD
        self.SELL_PERCENTAGE = SELL_PERCENTAGE
        self.TDFI_THRESHOLD = TDFI_THRESHOLD

        # Import the data
        try:
            self.df = pd.read_csv(f"data/BATS_{SYMBOL}.csv")
        except:
            self.df = pd.read_csv(f"Trading/data/BATS_{SYMBOL}.csv")
        self.df["time"] = pd.to_datetime(self.df["time"], unit="s")
        self.df = self.df[(self.df["time"] >= self.START_DATE) & (self.df["time"] <= self.END_DATE)]

        # Initialize EMA sequences
        self.df[f"EMA{EMA_FAST}"] = self.df["close"].ewm(halflife=f"{EMA_FAST/2} days", times=self.df["time"]).mean()
        self.df[f"EMA{EMA_SLOW}"] = self.df["close"].ewm(halflife=f"{EMA_SLOW/2} days", times=self.df["time"]).mean()
        self.df[f"EMA{EMA_SUPERSLOW}"] = self.df["close"].ewm(halflife=f"{EMA_SUPERSLOW/2} days", times=self.df["time"]).mean()

        self._run_strategy()
    
    def _run_strategy(self):
        strategy_capital = np.zeros(self.df.shape[0])
        strategy_capital[0] = self.INITIAL_CAPITAL
        strategy_position_size = np.zeros(self.df.shape[0])
        strategy_total_investment = np.zeros(self.df.shape[0])
        strategy_total_investment[0] = self.INITIAL_CAPITAL

        bah_position_size = np.zeros(self.df.shape[0])
        bah_position_size[0] = self.INITIAL_CAPITAL/self.df.iloc[0]["close"]

        annotations = np.full(self.df.shape[0], None)
        self._buy_annotations = np.full(self.df.shape[0], None)
        self._sell_annotations = np.full(self.df.shape[0], None)

        remaining_buy_time = 0
        buy_value = 0
        last_buy_order_price = self.df.iloc[0]["close"]

        for i_time in range(self.df.shape[0]-1):
            read_row = self.df.iloc[i_time]
            read_next_row = self.df.iloc[i_time+1]

            strategy_capital[i_time+1] = strategy_capital[i_time]
            strategy_total_investment[i_time+1] = strategy_total_investment[i_time]
            bah_position_size[i_time+1] = bah_position_size[i_time]
            strategy_position_size[i_time+1] = strategy_position_size[i_time]
            if i_time > 0 and read_row["time"].month != self.df.iloc[i_time-1]["time"].month:
                strategy_capital[i_time+1] += self.MONTHLY_INSTALLMENTS
                strategy_total_investment[i_time+1] += self.MONTHLY_INSTALLMENTS
                bah_position_size[i_time+1] += self.MONTHLY_INSTALLMENTS / read_next_row["open"]

            # sell whenever the RSI is above the threshold
            if read_row["RSI"] > self.RSI_THRESHOLD and read_row["close"] >= read_row[f"EMA{self.EMA_SUPERSLOW}"]:
                shares_to_sell = int(strategy_position_size[i_time] * self.SELL_PERCENTAGE)
                strategy_capital[i_time+1] += shares_to_sell * (read_next_row["open"] - self.COMMISSION_PER_SHARE_TRADED)
                strategy_position_size[i_time+1] -= shares_to_sell
                annotations[i_time] = f"-{shares_to_sell}"
                self._sell_annotations[i_time] = f"-{shares_to_sell}"
                continue

            # if the SHA delta becomes positive while the TDFI is over a threshold, go all in
            # if cross_under(np.zeros(self.df.shape[0]), self.df["Smoothed HA (Open)"] - self.df["Smoothed HA (Close)"], i_time) and read_row["TDFI"] > TDFI_THRESHOLD:
            #     remaining_buy_time = 1
            #     buy_value = strategy_capital[i_time]
            # while EMA_FAST > EMA_SLOW, buy aggressively
            if read_row[f"EMA{self.EMA_FAST}"] > read_row[f"EMA{self.EMA_SLOW}"]: # elif if you allow SHA
                if cross_under(self.df["close"], self.df[f"EMA{self.EMA_FAST}"], i_time):
                    remaining_buy_time = 5
                    buy_value = 0.02 * strategy_capital[i_time]
                if cross_under(self.df["close"], self.df[f"EMA{self.EMA_SLOW}"], i_time) or cross_under(self.df["close"], self.df[f"EMA{self.EMA_SUPERSLOW}"], i_time):
                    remaining_buy_time = 5
                    buy_value = 0.10 * strategy_capital[i_time]
            # while EMA_SLOW > EMA_FAST, buy less aggressively
            else:
                if cross_under(self.df["close"], self.df[f"EMA{self.EMA_SLOW}"], i_time):
                    remaining_buy_time = 5
                    buy_value = 0.05 * strategy_capital[i_time]
                if np.abs(read_row["close"] - last_buy_order_price)/last_buy_order_price > 0.10:
                    remaining_buy_time = 5
                    buy_value = 0.02 * strategy_capital[i_time]
            
            # if the close > EMA_FAST, abort any buys
            if read_row["close"] > read_row[f"EMA{self.EMA_FAST}"]:
                remaining_buy_time = 0
                buy_value = 0
            
            # resolve the buying
            if remaining_buy_time > 0:
                shares_to_buy = np.max(int(buy_value / (read_next_row["open"] + self.COMMISSION_PER_SHARE_TRADED)),0)
                strategy_capital[i_time+1] -= shares_to_buy * (read_next_row["open"] + self.COMMISSION_PER_SHARE_TRADED)
                strategy_position_size[i_time+1] += shares_to_buy
                annotations[i_time] = f"+{shares_to_buy}"
                self._buy_annotations[i_time] = f"+{shares_to_buy}"
                
                last_buy_order_price = read_next_row["open"]
                remaining_buy_time -= 1

        self.df["strategy_total_investment"] = strategy_total_investment
        self.df["strategy_capital"] = strategy_capital
        self.df["strategy_position_size"] = strategy_position_size
        self.df["strategy_equity"] = strategy_capital + strategy_position_size * self.df["close"]
        self.df["trade_annotations"] = annotations
        self.df["bah_equity"] = bah_position_size * self.df["close"]

    def candle_plot(self):
        # Plot the graphs
        fig = plotly.subplots.make_subplots(rows=2, cols=1, row_heights=[0.8, 0.2],  specs=[[{"secondary_y": True}],[{"secondary_y": True}]])
        fig.update_yaxes(fixedrange=False)

        # Plot price candlesticks
        fig.add_trace(plotly.graph_objects.Candlestick(
            x=self.df["time"],
            open=self.df["open"],
            high=self.df["high"],
            low=self.df["low"],
            close=self.df["close"],
            name="Price",
        ))

        # Buy/sell annotations
        fig.add_trace(plotly.graph_objects.Scatter(
            x=self.df["time"],
            y=self.df["high"],
            mode="text",
            text=self._buy_annotations,
            textposition="top center",
            textfont=dict(
                color="green",
                size=8
            ),
            showlegend=False,
        ))
        fig.add_trace(plotly.graph_objects.Scatter(
            x=self.df["time"],
            y=self.df["high"],
            mode="text",
            text=self._sell_annotations,
            textposition="top center",
            textfont=dict(
                color="red",
                size=8
            ),
            showlegend=False,
        ))

        # Plot smoothed HA
        fig.add_trace(plotly.graph_objects.Candlestick(
            x=self.df["time"],
            open=self.df["Smoothed HA (Open)"],
            high=self.df["Smoothed HA (High)"],
            low=self.df["Smoothed HA (Low)"],
            close=self.df["Smoothed HA (Close)"],
            increasing_line_color= "lightgray",
            decreasing_line_color= "pink",
            name="Smoothed HA"
        ))

        # Plot EMA lines
        fig.add_trace(plotly.graph_objects.Scatter(
            x=self.df["time"],
            y=self.df[f"EMA{self.EMA_FAST}"],
            name=f"EMA {self.EMA_FAST}",
            line=dict(color="red", dash="dot")
        ))
        fig.add_trace(plotly.graph_objects.Scatter(
            x=self.df["time"],
            y=self.df[f"EMA{self.EMA_SLOW}"],
            name=f"EMA {self.EMA_SLOW}",
            line=dict(color="orange", dash="dot")
        ))
        fig.add_trace(plotly.graph_objects.Scatter(
            x=self.df["time"],
            y=self.df[f"EMA{self.EMA_SUPERSLOW}"],
            name=f"EMA {self.EMA_SUPERSLOW}",
            line=dict(color="gold", dash="dot")
        ))
        # fig.add_trace(plotly.graph_objects.Scatter( # EMA 200
        #     x=self.df["time"],
        #     y=self.df["close"].ewm(halflife=f"{200/2} days", times=self.df["time"]).mean(),
        #     name=f"EMA 200",
        #     line=dict(color="green", dash="dot")
        # ))

        # Plot TDFI and RSI
        fig.add_trace(plotly.graph_objects.Scatter(
            x=self.df["time"],
            y=self.df["TDFI"],
            name="TDFI",
        ), row=2, col=1)
        fig.add_trace(plotly.graph_objects.Scatter(
            x=self.df["time"],
            y=self.df["RSI"],
            name="RSI",
        ), row=2, col=1,
        secondary_y=True)

        fig.update_layout(xaxis_rangeslider_visible=False, autosize=False, width=1000, height=800)
        fig.show()

    def display_performance(self):
        # Print results summary
        print("STRATEGY RESULTS")
        print("Final equity as percentage of investment: %.3f%%" % (self.df["strategy_equity"].iloc[-1]/self.df["strategy_total_investment"].iloc[-1]*100))
        effann = (self.df["strategy_equity"].iloc[-1]/self.df["strategy_total_investment"].iloc[-1]) ** (1/((self.END_DATE - self.START_DATE).days/365.25)) * 100 -100
        print("Equivalent annual growth rate: %s%.3f%%" % (
            "+" if effann >= 0 else "",
            effann
        ))
        print("Total investment:", self.df["strategy_total_investment"].iloc[-1])
        print("Total final equity:", self.df["strategy_equity"].iloc[-1])

        print()
        print("BUY AND HOLD COMPARISON")
        print("Final equity as percentage of investment: %.3f%%" % (self.df["bah_equity"].iloc[-1]/self.df["strategy_total_investment"].iloc[-1]*100))
        effann = (self.df["bah_equity"].iloc[-1]/self.df["strategy_total_investment"].iloc[-1]) ** (1/((self.END_DATE - self.START_DATE).days/365.25)) * 100 -100
        print("Equivalent annual growth rate: %s%.3f%%" % (
            "+" if effann >= 0 else "",
            effann
        ))
        print("Total investment:", self.df["strategy_total_investment"].iloc[-1])
        print("Total final equity:", self.df["bah_equity"].iloc[-1])

        # Plot results
        plt.plot(self.df["time"], self.df["bah_equity"]/self.INITIAL_CAPITAL*100, label="Buy and Hold", color="lightgray", linewidth=1)
        plt.plot(self.df["time"], self.df["strategy_equity"]/self.INITIAL_CAPITAL*100, label="Strategy", color="black", linewidth=1)
        plt.plot(self.df["time"], self.df["strategy_total_investment"]/self.INITIAL_CAPITAL*100, label="Total Investment", color="lightgreen", linewidth=1)
        plt.xlabel("Time")
        plt.ylabel("Total equity (%)")
        plt.legend()
        plt.show()

    def perf_table(self):
        # Display action table
        return self.df[["time", "close", "strategy_capital", "strategy_position_size", "strategy_equity", "trade_annotations"]].style.hide()
import backtrader as bt
import logging

# Custom Momentum Indicator: Rate of Change
from custom_roc import ROC
# Custom Sizer: Risk 2% of cash per trade
class RiskSizer(bt.Sizer):
    params = (('risk', 0.02),)

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            risk_cash = cash * self.p.risk
            price = data.close[0]
            size = int(risk_cash // price)
            return max(size, 0)
        else:
            return self.broker.getposition(data).size

# Main Strategy
from custom_risk_sizer import RiskSizer
        self.log_list = []
        logging.basicConfig(filename='coding/msft_momentum_logs.txt', level=logging.INFO)

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        log_entry = f'{dt.isoformat()}, {txt}'
        self.log_list.append(log_entry)
        logging.info(log_entry)

    def next(self):
        if self.order:
            return  # waiting for previous order

        pos = self.getposition()
        if not pos:
            # Entry condition: ROC crosses above threshold and price > SMA
            if self.roc[0] > self.p.roc_buy and self.data.close[0] > self.sma[0]:
                self.log(f'BUY CREATE, Price: {self.data.close[0]:.2f}, ROC: {self.roc[0]:.2f}')
                self.order = self.buy()
        else:
            # Exit: ROC falls below threshold or price < SMA
            if self.roc[0] < self.p.roc_sell or self.data.close[0] < self.sma[0]:
                self.log(f'SELL CREATE, Price: {self.data.close[0]:.2f}, ROC: {self.roc[0]:.2f}')
                self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order = None
        if order.status == order.Completed:
            self.log(f'ORDER EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size}')

    def stop(self):
        self.log(f'FINAL PORTFOLIO VALUE: {self.broker.getvalue():.2f}')

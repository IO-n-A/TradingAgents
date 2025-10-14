import backtrader as bt

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

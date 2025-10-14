import backtrader as bt

class ROC(bt.Indicator):
    lines = ('roc',)
    params = (('period', 20),)

    def __init__(self):
        self.addminperiod(self.p.period + 1)

    def next(self):
        prev_close = self.data.close[-self.p.period]
        if prev_close != 0:
            self.lines.roc[0] = 100.0 * (self.data.close[0] - prev_close) / prev_close
        else:
            self.lines.roc[0] = 0.0
